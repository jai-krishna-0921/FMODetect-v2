"""Inference utilities: run FMODetect-v2 on a single image or a video.

Mirrors the contract of the original run.py: takes an image + background OR a
video, returns the TDF heatmap, the H*M matting prediction, and a colour overlay
suitable for display.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..models.unet import FMODetectNet, UNetConfig
from .trajectory import Trajectory, extract_trajectories, overlay_trajectories, trajectory_as_dict


def _pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad H,W up to the next multiple. Returns padded array and (orig_h, orig_w)."""
    h, w = arr.shape[:2]
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    if nh == h and nw == w:
        return arr, (h, w)
    pad = np.zeros((nh, nw, arr.shape[2]) if arr.ndim == 3 else (nh, nw), dtype=arr.dtype)
    pad[:h, :w] = arr
    return pad, (h, w)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


@dataclass
class InferOutputs:
    tdf: np.ndarray            # (H, W) in [0, 1]
    hm: np.ndarray | None      # (H, W) in [0, 1] or None if matting head disabled
    log_var: np.ndarray | None # (H, W) per-pixel log variance (if uncertainty head on)
    overlay: np.ndarray        # (H, W, 3) uint8 BGR display overlay
    trajectories: list[Trajectory]  # post-hoc parametric trajectories


def colorize_tdf(tdf: np.ndarray, image_rgb01: np.ndarray) -> np.ndarray:
    """Red trajectory on top of grayscale image."""
    base = (image_rgb01.mean(-1) * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    heat = (np.clip(tdf, 0, 1) * 255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(base_rgb, 0.55, heat_col, 0.45, 0)


def load_model(ckpt_path: Path, device: torch.device | str = "cuda") -> FMODetectNet:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = UNetConfig()
    model = FMODetectNet(cfg)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval().to(device)
    return model


@torch.no_grad()
def infer_pair(model: FMODetectNet, image_rgb01: np.ndarray, bgr_rgb01: np.ndarray,
               device: torch.device | str = "cuda",
               *, extract_traj: bool = True, tdf_threshold: float = 0.5) -> InferOutputs:
    """Run one image + background pair through the net + post-hoc trajectory fit."""
    assert image_rgb01.shape == bgr_rgb01.shape, "image and background must share shape"
    img = _normalize(image_rgb01.astype(np.float32))
    bg = _normalize(bgr_rgb01.astype(np.float32))
    x = np.concatenate([img, bg], axis=-1)
    x, (h0, w0) = _pad_to_multiple(x, 16)
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    out = model(t)
    tdf = out["tdf"][0, 0, :h0, :w0].float().cpu().numpy()
    hm = out["hm"][0, 0, :h0, :w0].float().cpu().numpy() if "hm" in out else None
    lv = out["log_var_tdf"][0, 0, :h0, :w0].float().cpu().numpy() if "log_var_tdf" in out else None

    trajs = extract_trajectories(tdf, tdf_threshold=tdf_threshold) if extract_traj else []
    if trajs:
        overlay = overlay_trajectories(image_rgb01, trajs)
    else:
        overlay = colorize_tdf(tdf, image_rgb01)
    return InferOutputs(tdf=tdf, hm=hm, log_var=lv, overlay=overlay, trajectories=trajs)


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(frames, axis=0), axis=0)


def infer_video(model: FMODetectNet, video_path: Path, out_path: Path,
                median_window: int = 3, device: torch.device | str = "cuda") -> dict:
    """Process every frame with a running-median background. Writes MP4 overlay."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    buf: list[np.ndarray] = []
    writer: cv2.VideoWriter | None = None
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        buf.append(rgb)
        if len(buf) > median_window:
            buf.pop(0)
        bg = median_background(buf)
        res = infer_pair(model, rgb, bg, device=device)
        if writer is None:
            h, w = res.overlay.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        writer.write(res.overlay)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    return {"frames": n, "out": str(out_path)}
