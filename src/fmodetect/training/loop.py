"""4GB-aware training loop with AMP, grad accumulation, MLflow + (optional) ClearML."""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.dataset import SynthFMODataset
from ..models.losses import combined_loss
from ..models.unet import FMODetectNet, UNetConfig
from ..utils.config import Config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_dataset(ds: SynthFMODataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    n = len(ds)
    n_val = max(1, int(round(n * val_fraction)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    ds = SynthFMODataset(cfg.data.h5_path)
    train_set, val_set = _split_dataset(ds, cfg.data.val_fraction, cfg.seed)
    common = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def _build_model(cfg: Config) -> FMODetectNet:
    return FMODetectNet(UNetConfig(
        in_channels=cfg.model.in_channels,
        base_channels=tuple(cfg.model.base_channels),  # type: ignore[arg-type]
        use_cbam=cfg.model.use_cbam,
        predict_matting=cfg.model.predict_matting,
        predict_uncertainty=cfg.model.predict_uncertainty,
    ))


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device,
                    channels_last: bool) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        t = v.to(device, non_blocking=True)
        if channels_last and t.dim() == 4:
            t = t.contiguous(memory_format=torch.channels_last)
        out[k] = t
    return out


@torch.no_grad()
def _validate(model: FMODetectNet, loader: DataLoader, device: torch.device,
              cfg: Config) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move_to_device(batch, device, cfg.train.channels_last)
        with torch.autocast(device_type=device.type, enabled=cfg.train.amp, dtype=torch.float16):
            out = model(batch["x"])
            parts = combined_loss(out, {"tdf": batch["tdf"], "hm": batch["hm"]},
                                  matting_weight=cfg.loss.matting_weight,
                                  boundary_weight=cfg.loss.boundary_weight)
        bs = batch["x"].size(0)
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + float(v.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def _save_ckpt(model: FMODetectNet, opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
               epoch: int, best_loss: float, ckpt_dir: Path, tag: str) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"{tag}.pt"
    torch.save({
        "epoch": epoch,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
    }, p)
    return p


def train(cfg: Config) -> None:
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    train_loader, val_loader = _build_loaders(cfg)
    print(f"[train] train batches={len(train_loader)}, val batches={len(val_loader)}")

    model = _build_model(cfg).to(device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")

    # MLflow
    mlflow.set_tracking_uri(cfg.logging.mlflow_uri)
    mlflow.set_experiment(cfg.logging.mlflow_experiment)
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        **{f"model.{k}": v for k, v in cfg.model.model_dump().items()},
        **{f"train.{k}": v for k, v in cfg.train.model_dump().items()},
        **{f"loss.{k}": v for k, v in cfg.loss.model_dump().items()},
        "data.h5_path": cfg.data.h5_path,
        "params_M": n_params / 1e6,
    })

    # Optional ClearML mirror
    if cfg.logging.use_clearml:
        try:
            from clearml import Task
            Task.init(project_name=cfg.logging.clearml_project, task_name=run_name)
        except Exception as e:  # noqa: BLE001
            print(f"[train] clearml init skipped: {e}")

    tb = SummaryWriter(log_dir=str(Path(cfg.logging.tensorboard_dir) / run_name))
    ckpt_dir = Path(cfg.checkpoints.dir) / run_name

    best_loss = float("inf")
    patience_left = cfg.train.early_stop_patience
    global_step = 0
    accum = cfg.train.grad_accum_steps

    for epoch in range(cfg.train.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        running: dict[str, float] = {}
        n_seen = 0
        for step, batch in enumerate(pbar):
            batch = _move_to_device(batch, device, cfg.train.channels_last)
            with torch.autocast(device_type=device.type, enabled=cfg.train.amp, dtype=torch.float16):
                out = model(batch["x"])
                parts = combined_loss(out, {"tdf": batch["tdf"], "hm": batch["hm"]},
                                      matting_weight=cfg.loss.matting_weight,
                                      boundary_weight=cfg.loss.boundary_weight)
                loss = parts["total"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                if cfg.train.grad_clip_norm:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            bs = batch["x"].size(0)
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v.item()) * bs
            n_seen += bs
            if step % cfg.train.log_every_n_steps == 0:
                pbar.set_postfix(total=f"{parts['total'].item():.4f}")

        # epoch averages
        train_avg = {f"train/{k}": v / n_seen for k, v in running.items()}
        for k, v in train_avg.items():
            tb.add_scalar(k, v, epoch)
            mlflow.log_metric(k, v, step=epoch)

        if (epoch + 1) % cfg.train.val_every_n_epochs == 0:
            val_avg = _validate(model, val_loader, device, cfg)
            for k, v in val_avg.items():
                tb.add_scalar(f"val/{k}", v, epoch)
                mlflow.log_metric(f"val/{k}", v, step=epoch)
            cur_val = val_avg.get("total", float("inf"))
            print(f"[train] epoch {epoch}: train_total={train_avg.get('train/total'):.4f} "
                  f"val_total={cur_val:.4f}")
            improved = cur_val < best_loss - 1e-5
            if improved:
                best_loss = cur_val
                patience_left = cfg.train.early_stop_patience
                p = _save_ckpt(model, opt, scaler, epoch, best_loss, ckpt_dir, "best")
                mlflow.log_artifact(str(p))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[train] early stopping at epoch {epoch}")
                    break

        if (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            _save_ckpt(model, opt, scaler, epoch, best_loss, ckpt_dir, f"epoch_{epoch:03d}")

    _save_ckpt(model, opt, scaler, epoch, best_loss, ckpt_dir, "last")
    mlflow.end_run()
    tb.close()
    print(f"[train] done. best val loss = {best_loss:.5f}")
