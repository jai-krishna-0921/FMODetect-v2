"""End-to-end eval script.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/eval.py \\
        --ckpt experiments/checkpoints/<run>/best.pt \\
        --dataset datasets/eval/TbD \\
        --out experiments/eval/tbd_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.fmodetect.eval.datasets import EvalFrame, load_eval_dataset
from src.fmodetect.eval.metrics import bbox_iou, precision_recall, trajectory_iou
from src.fmodetect.inference.runner import infer_pair, load_model
from src.fmodetect.inference.trajectory import extract_trajectories


def evaluate_dataset(model, frames: list[EvalFrame], device: str = "cuda",
                     *, iou_tp_th: float = 0.1) -> dict:
    tp = fp = fn = 0
    tious: list[float] = []
    per_frame: list[dict] = []
    n_with_gt = sum(1 for f in frames if f.gt_traj_yx is not None)

    for f in tqdm(frames, desc=f"eval {frames[0].sequence}"):
        img = cv2.cvtColor(cv2.imread(str(f.image_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if f.background_path is None:
            # Fall back: per-frame median over a 3-frame causal window is not available here,
            # so use a per-channel mean as a weak background.
            bg = np.broadcast_to(img.mean(axis=(0, 1)), img.shape).astype(np.float32)
        else:
            bg = cv2.cvtColor(cv2.imread(str(f.background_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if bg.shape != img.shape:
                bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
        res = infer_pair(model, img, bg, device=device)
        trajs = extract_trajectories(res.tdf)

        gt_present = f.gt_traj_yx is not None and f.gt_bbox_yxyx is not None
        # Detection P/R via bbox IoU vs GT bbox
        if gt_present:
            matched = False
            for t in trajs:
                if bbox_iou(t.bbox, f.gt_bbox_yxyx) >= iou_tp_th:
                    matched = True
                    break
            if matched:
                tp += 1
            else:
                fn += 1
            # Unmatched predictions count as FP
            fp += max(0, len(trajs) - (1 if matched else 0))
            # TIoU on the (best-IoU) match if present
            if matched and trajs:
                best = max(trajs, key=lambda t: bbox_iou(t.bbox, f.gt_bbox_yxyx))
                tiou = trajectory_iou(f.gt_traj_yx, best.points, f.gt_radius_px or 5.0, img.shape[:2])
                tious.append(tiou)
        else:
            fp += len(trajs)

        per_frame.append({
            "frame": f.frame_idx, "image": str(f.image_path),
            "n_predicted": len(trajs),
            "has_gt": gt_present,
        })

    p, r = precision_recall(tp, fp, fn)
    return {
        "sequence": frames[0].sequence if frames else None,
        "n_frames": len(frames),
        "n_frames_with_gt": n_with_gt,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r,
        "mean_tiou": float(np.mean(tious)) if tious else None,
        "median_tiou": float(np.median(tious)) if tious else None,
        "per_frame": per_frame[:50],  # truncate to first 50 for log readability
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    sequences = load_eval_dataset(args.dataset)
    if not sequences:
        raise SystemExit(f"no usable sequences under {args.dataset}")
    print(f"[eval] {len(sequences)} sequences, "
          f"{sum(len(v) for v in sequences.values())} frames")

    results = {"dataset": str(args.dataset), "ckpt": str(args.ckpt), "sequences": {}}
    t0 = perf_counter()
    all_tp = all_fp = all_fn = 0
    all_tious: list[float] = []
    for name, frames in sequences.items():
        r = evaluate_dataset(model, frames, device=str(device))
        results["sequences"][name] = r
        all_tp += r["tp"]; all_fp += r["fp"]; all_fn += r["fn"]
        if r["mean_tiou"] is not None:
            all_tious.append(r["mean_tiou"])
    p, recall = precision_recall(all_tp, all_fp, all_fn)
    results["aggregate"] = {
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "precision": p, "recall": recall,
        "mean_tiou": float(np.mean(all_tious)) if all_tious else None,
        "wall_time_s": perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    agg = results["aggregate"]
    print(f"[eval] precision={agg['precision']:.3f} recall={agg['recall']:.3f} "
          f"TIoU={agg['mean_tiou']} ({agg['wall_time_s']:.1f}s)")


if __name__ == "__main__":
    main()
