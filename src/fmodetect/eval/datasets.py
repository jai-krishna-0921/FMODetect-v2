"""Loaders for the three real-world FMO eval sets.

Each loader yields per-frame dicts:
    {
        'sequence': str,
        'frame_idx': int,
        'image_path': Path,        # full RGB frame
        'background_path': Path | None,
        'gt_traj_yx': (N, 2) np.ndarray | None,   # ordered trajectory points
        'gt_radius_px': float | None,
        'gt_bbox_yxyx': tuple | None,             # (y0, x0, y1, x1)
    }

Annotation format reference (per the paper supplementary + repo conventions):
  TbD / TbD-3D / falling — directory per sequence containing:
      gt/<n>.png    (per-frame trajectory mask, white pixels on black)
      <n>.png       (RGB frame)
      bgr.png       (median background)
      template.png  (object template for radius estimation)
The exact layout we observe is verified at runtime by scanning the directory
tree; this module is lenient and skips sequences without the expected files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class EvalFrame:
    sequence: str
    frame_idx: int
    image_path: Path
    background_path: Path | None
    gt_traj_yx: np.ndarray | None
    gt_radius_px: float | None
    gt_bbox_yxyx: tuple[int, int, int, int] | None


def _traj_from_mask(mask_path: Path) -> tuple[np.ndarray | None, float | None, tuple | None]:
    """Convert a GT trajectory mask PNG into an ordered (y, x) list + a radius estimate."""
    if not mask_path.exists():
        return None, None, None
    m = np.asarray(Image.open(mask_path).convert("L"))
    bin_m = m > 30
    if not bin_m.any():
        return None, None, None
    ys, xs = np.where(bin_m)
    pts = np.column_stack([ys, xs])
    # Radius estimate from connected-component extent / line-thickness:
    # use min(half_height, half_width) of bbox of one isolated blob if present;
    # else fall back to 2.0 (placeholder).
    y0, x0 = ys.min(), xs.min()
    y1, x1 = ys.max() + 1, xs.max() + 1
    # crude thickness: area / max(extent)
    area = bin_m.sum()
    extent = max(y1 - y0, x1 - x0)
    rad = max(2.0, float(area) / max(1.0, float(extent)) / 2.0)
    return pts, rad, (int(y0), int(x0), int(y1), int(x1))


def iter_sequence_dir(seq_root: Path) -> list[EvalFrame]:
    """Yield frames for a single eval sequence. Tolerant of varying layouts."""
    out: list[EvalFrame] = []
    # Strategy: every PNG/JPG in seq_root (not in gt/) is treated as a frame;
    # corresponding GT lives at gt/<stem>.png if present.
    bg_candidates = [seq_root / "bgr.png", seq_root / "background.png"]
    bg_path = next((p for p in bg_candidates if p.exists()), None)
    frames = sorted([p for p in seq_root.iterdir()
                     if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                     and p.name not in {"bgr.png", "background.png", "template.png"}])
    for idx, fp in enumerate(frames):
        gt_path = seq_root / "gt" / f"{fp.stem}.png"
        traj, rad, bbox = _traj_from_mask(gt_path)
        out.append(EvalFrame(
            sequence=seq_root.name,
            frame_idx=idx,
            image_path=fp,
            background_path=bg_path,
            gt_traj_yx=traj,
            gt_radius_px=rad,
            gt_bbox_yxyx=bbox,
        ))
    return out


def load_eval_dataset(root: Path) -> dict[str, list[EvalFrame]]:
    """Return {sequence_name: [EvalFrame, ...]} for a dataset root."""
    if not root.exists():
        return {}
    sequences = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in {"_logs"}:
            frames = iter_sequence_dir(child)
            if frames:
                sequences[child.name] = frames
    return sequences


def quick_summary(root: Path) -> dict:
    """Lightweight inspection — does this dataset look usable?"""
    seqs = load_eval_dataset(root)
    n_with_gt = 0
    total_frames = 0
    for frames in seqs.values():
        total_frames += len(frames)
        n_with_gt += sum(1 for f in frames if f.gt_traj_yx is not None)
    return {
        "root": str(root),
        "exists": root.exists(),
        "n_sequences": len(seqs),
        "n_frames": total_frames,
        "n_frames_with_gt": n_with_gt,
    }


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(quick_summary(args.root), indent=2))
