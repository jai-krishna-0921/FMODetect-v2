"""Loaders for the real-world FMO eval sets.

Each dataset has a different GT format. We expose one EvalFrame schema and route
to the right parser based on auto-detected directory layout.

Layouts observed in the actual downloads:

- **falling** (ptak.felk.cvut.cz/personal/rozumden/falling_imgs_gt.zip):
    falling/
        imgs/<seq>/00000000.png ...        # frames
        imgs_gt/<seq>/...                  # frames with GT drawn
        gt_bbox/<seq>.txt                  # "x y w h" per line (one per frame)
        roi_frames.txt                     # "first last" pairs (only FMO-present frames)

- **TbD / TbD-3D** (similar to falling but per-sequence dirs at the top level):
    TbD/<seq>/
        imgs/...      or 00000000.png ...
        gt/...        per-frame mask or per-frame curve params
        template.png  object template
        bgr.png       median background

- **FMO 2017** (cmp.felk.cvut.cz/fmo/files/fmo-cpp-experiment-2017-05-26.zip
                + gt-fmo-txt-2017-05-26.zip):
    FMO/sequences/<seq>.avi          # video
    FMO/gt_v1/<seq>_gt.txt           # text GT in pixel-index format
        line 1: "W H F L"
        lines 2..: "I N idx1 idx2 ... idxN"  (1-based pixel indices)
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
    gt_traj_yx: np.ndarray | None         # (N, 2) (y, x) pixel coords of trajectory
    gt_radius_px: float | None
    gt_bbox_yxyx: tuple[int, int, int, int] | None
    # Source kind for diagnostics: 'bbox', 'mask', 'pixel-idx', or None
    gt_kind: str | None = None


# -----------------------------------------------------------------------------
# Format-specific parsers
# -----------------------------------------------------------------------------

def _traj_from_mask_png(mask_path: Path) -> tuple[np.ndarray | None, float | None, tuple | None]:
    """GT is a PNG with trajectory drawn as white pixels on black."""
    if not mask_path.exists():
        return None, None, None
    m = np.asarray(Image.open(mask_path).convert("L"))
    bin_m = m > 30
    if not bin_m.any():
        return None, None, None
    ys, xs = np.where(bin_m)
    pts = np.column_stack([ys, xs])
    y0, x0 = int(ys.min()), int(xs.min())
    y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
    area = int(bin_m.sum())
    extent = max(y1 - y0, x1 - x0)
    rad = max(2.0, float(area) / max(1.0, float(extent)) / 2.0)
    return pts, rad, (y0, x0, y1, x1)


def _parse_bbox_file(bbox_path: Path) -> list[tuple[float, float, float, float]]:
    """Parse 'x y w h' bbox-per-frame file. Returns list of (x, y, w, h)."""
    rows: list[tuple[float, float, float, float]] = []
    with bbox_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, w, h = (float(p) for p in parts[:4])
                rows.append((x, y, w, h))
    return rows


def _gt_from_bbox(x: float, y: float, w: float, h: float) -> tuple[np.ndarray, float, tuple]:
    """Convert (x,y,w,h) bbox to a single-point 'trajectory' (center) + radius (half-min-dim)."""
    cy = y + h / 2.0
    cx = x + w / 2.0
    rad = max(2.0, min(w, h) / 2.0)
    bbox_yxyx = (int(y), int(x), int(y + h), int(x + w))
    return np.array([[cy, cx]], dtype=np.float32), rad, bbox_yxyx


def _parse_fmo_txt_gt(gt_path: Path, w: int, h: int) -> dict[int, np.ndarray]:
    """Parse the FMO 2017 text GT: returns {frame_idx (0-based): (N,2) yx array}.

    File format:
        W H F L
        I N idx1 idx2 ... idxN
        ...
    Pixel index i (1-based) → (x = (i-1) div H, y = (i-1) mod H).
    """
    out: dict[int, np.ndarray] = {}
    with gt_path.open() as f:
        first = f.readline().strip().split()
        # First line: W H F L
        if len(first) < 4:
            return out
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            frame = int(parts[0]) - 1            # 0-based
            n = int(parts[1])
            idxs = np.array([int(p) for p in parts[2:2 + n]], dtype=np.int64) - 1
            xs = idxs // h
            ys = idxs % h
            out[frame] = np.column_stack([ys, xs]).astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# Dataset-specific iteration
# -----------------------------------------------------------------------------

def iter_falling_sequence(seq_dir: Path, gt_bbox_path: Path | None) -> list[EvalFrame]:
    """Falling layout: imgs/<seq>/{frames}.png + gt_bbox/<seq>.txt."""
    frames = sorted(seq_dir.glob("*.png"))
    bboxes = _parse_bbox_file(gt_bbox_path) if gt_bbox_path and gt_bbox_path.exists() else []
    out: list[EvalFrame] = []
    for idx, fp in enumerate(frames):
        if idx < len(bboxes):
            traj, rad, bbox = _gt_from_bbox(*bboxes[idx])
            out.append(EvalFrame(
                sequence=seq_dir.name, frame_idx=idx, image_path=fp,
                background_path=None, gt_traj_yx=traj, gt_radius_px=rad,
                gt_bbox_yxyx=bbox, gt_kind="bbox",
            ))
        else:
            out.append(EvalFrame(
                sequence=seq_dir.name, frame_idx=idx, image_path=fp,
                background_path=None, gt_traj_yx=None, gt_radius_px=None,
                gt_bbox_yxyx=None, gt_kind=None,
            ))
    return out


def iter_tbd_sequence(seq_dir: Path) -> list[EvalFrame]:
    """TbD / TbD-3D layout (best-effort, varies by sub-version)."""
    # bg/template at sequence root
    bg = next((seq_dir / n for n in ("bgr.png", "background.png")
               if (seq_dir / n).exists()), None)
    # frames may be in seq_dir/ or seq_dir/imgs/
    img_dir = seq_dir / "imgs" if (seq_dir / "imgs").is_dir() else seq_dir
    frames = sorted(p for p in img_dir.glob("*.png")
                    if p.name not in {"bgr.png", "background.png", "template.png"})
    gt_dir = seq_dir / "gt"
    out: list[EvalFrame] = []
    for idx, fp in enumerate(frames):
        gt_png = gt_dir / f"{fp.stem}.png"
        if gt_png.exists():
            traj, rad, bbox = _traj_from_mask_png(gt_png)
            kind = "mask" if traj is not None else None
        else:
            traj, rad, bbox, kind = None, None, None, None
        out.append(EvalFrame(
            sequence=seq_dir.name, frame_idx=idx, image_path=fp,
            background_path=bg, gt_traj_yx=traj, gt_radius_px=rad,
            gt_bbox_yxyx=bbox, gt_kind=kind,
        ))
    return out


def load_falling(root: Path) -> dict[str, list[EvalFrame]]:
    img_root = root / "imgs"
    gt_root = root / "gt_bbox"
    if not img_root.is_dir():
        return {}
    sequences: dict[str, list[EvalFrame]] = {}
    for seq_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        gt_path = gt_root / f"{seq_dir.name}.txt"
        sequences[seq_dir.name] = iter_falling_sequence(seq_dir, gt_path)
    return sequences


def load_tbd_like(root: Path) -> dict[str, list[EvalFrame]]:
    """TbD / TbD-3D dataset loader.

    Two observed layouts:
      (a) Sequences at top level: TbD/<seq>/*.png
      (b) Wrapped under imgs/ (TbD-3D actually): TbD-3D/imgs/<seq>/*.png
    """
    if not root.is_dir():
        return {}
    # If there's an imgs/ dir at the top, use it as the sequence root
    if (root / "imgs").is_dir():
        seq_root = root / "imgs"
    else:
        seq_root = root
    sequences: dict[str, list[EvalFrame]] = {}
    for child in sorted(p for p in seq_root.iterdir() if p.is_dir() and p.name not in {"_logs"}):
        frames = iter_tbd_sequence(child)
        if frames:
            sequences[child.name] = frames
    return sequences


def load_fmo2017(root: Path) -> dict[str, list[EvalFrame]]:
    """FMO 2017 — sequences are .avi videos + .txt pixel-index GT."""
    # Best effort: locate .avi videos and matching gt_v1 / gt_v2 .txt files.
    if not root.is_dir():
        return {}
    seq_paths = sorted(list(root.rglob("*.avi")))
    sequences: dict[str, list[EvalFrame]] = {}
    for avi in seq_paths:
        stem = avi.stem
        gt_txt = next((p for p in [root / "gt_v2" / f"{stem}_gt.txt",
                                   root / "gt_v1" / f"{stem}_gt.txt",
                                   root / f"{stem}_gt.txt"] if p.exists()), None)
        if gt_txt is None:
            continue
        # We don't decode video frames here. Caller is expected to use OpenCV.
        # Stash as a single EvalFrame with image_path=video (frame_idx left at 0).
        # An evaluation runner can iterate cv2.VideoCapture(avi) and look up GT by frame.
        sequences[stem] = [EvalFrame(
            sequence=stem, frame_idx=-1, image_path=avi, background_path=None,
            gt_traj_yx=None, gt_radius_px=None, gt_bbox_yxyx=None,
            gt_kind="pixel-idx-video",
        )]
    return sequences


def load_eval_dataset(root: Path) -> dict[str, list[EvalFrame]]:
    """Auto-detect format from directory layout, return {seq: [EvalFrame, ...]}."""
    if not root.exists():
        return {}
    # Detect: falling has 'imgs/' + 'gt_bbox/' at top level.
    if (root / "imgs").is_dir() and (root / "gt_bbox").is_dir():
        return load_falling(root)
    # Detect: FMO 2017 has .avi files anywhere under the tree.
    if any(root.rglob("*.avi")):
        return load_fmo2017(root)
    # Default: TbD-style.
    return load_tbd_like(root)


def quick_summary(root: Path) -> dict:
    seqs = load_eval_dataset(root)
    n_with_gt = 0
    total = 0
    kinds: dict[str, int] = {}
    for frames in seqs.values():
        total += len(frames)
        for f in frames:
            if f.gt_traj_yx is not None or f.gt_kind == "pixel-idx-video":
                n_with_gt += 1
            if f.gt_kind:
                kinds[f.gt_kind] = kinds.get(f.gt_kind, 0) + 1
    return {
        "root": str(root),
        "exists": root.exists(),
        "n_sequences": len(seqs),
        "n_frames": total,
        "n_frames_with_gt": n_with_gt,
        "gt_kinds": kinds,
    }


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(quick_summary(args.root), indent=2))
