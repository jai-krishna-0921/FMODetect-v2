"""Run eval on the falling dataset + generate visual overlay images.

For each frame:
  - run model inference
  - extract trajectories
  - draw on the frame: GT bbox (green) + predicted trajectory (orange) + speed/conf label
  - save to experiments/v2_falling_eval/<seq>/<frame>.png

Also writes:
  - experiments/v2_falling_eval/_metrics.json   (per-seq + aggregate)
  - experiments/v2_falling_eval/_grid.png       (one row per sequence, 4 samples each)
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.fmodetect.eval.datasets import load_falling
from src.fmodetect.eval.metrics import bbox_iou, precision_recall, trajectory_iou
from src.fmodetect.inference.runner import load_model, infer_pair


def draw_annotated(img_bgr: np.ndarray, gt_bbox, trajs) -> np.ndarray:
    """Draw GT bbox (green) + predicted trajectories (orange line + endpoints + label)."""
    out = img_bgr.copy()
    # GT bbox: (y0, x0, y1, x1)
    if gt_bbox is not None:
        y0, x0, y1, x1 = gt_bbox
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, "GT", (x0, max(0, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Predicted trajectories
    for i, t in enumerate(trajs):
        n = 64
        ts = np.linspace(0.0, 1.0, n)
        a = np.minimum(2 * ts, 1.0)
        b = np.maximum(2 * ts - 1.0, 0.0)
        y = t.curve_params[0, 0] + t.curve_params[0, 1] * a + t.curve_params[0, 2] * a * a + t.curve_params[0, 3] * b
        x = t.curve_params[1, 0] + t.curve_params[1, 1] * a + t.curve_params[1, 2] * a * a + t.curve_params[1, 3] * b
        pts = np.column_stack([x, y]).astype(np.int32)
        cv2.polylines(out, [pts], False, (0, 165, 255), 3)
        cv2.circle(out, (int(x[0]), int(y[0])), 6, (0, 200, 0), -1)
        cv2.circle(out, (int(x[-1]), int(y[-1])), 6, (0, 0, 255), -1)
        # bbox for pred
        py0, px0, py1, px1 = t.bbox
        cv2.rectangle(out, (px0, py0), (px1, py1), (0, 165, 255), 1)
        # label
        label = f"#{i+1} v={t.speed_px_per_frame:.0f}px/f r={t.radius_px:.0f} c={t.confidence:.2f}"
        cv2.putText(out, label, (px0, max(15, py0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/eval/falling"))
    ap.add_argument("--out", type=Path, default=Path("experiments/v2_falling_eval"))
    ap.add_argument("--tdf-thr", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=5,
                    help="Save every N-th annotated frame (to keep disk modest).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.ckpt} on {device}")
    model = load_model(args.ckpt, device=device)

    seqs = load_falling(args.dataset)
    print(f"{len(seqs)} sequences, {sum(len(v) for v in seqs.values())} frames")

    grid_samples: list[np.ndarray] = []  # one row of 4 frames per sequence
    per_seq: dict[str, dict] = {}
    all_tp = all_fp = all_fn = 0
    all_tious: list[float] = []

    for name, frames in seqs.items():
        seq_dir = args.out / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        tp = fp = fn = 0
        tious: list[float] = []
        saved_grid_imgs = []
        # Compute one background per sequence: median over a stride of frames
        bg_stack = []
        for f in frames[::max(1, len(frames)//8)][:8]:
            im = cv2.cvtColor(cv2.imread(str(f.image_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            bg_stack.append(im)
        bg_per_seq = np.median(np.stack(bg_stack), axis=0).astype(np.float32)
        for f in tqdm(frames, desc=name):
            img_bgr = cv2.imread(str(f.image_path))
            img_rgb01 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            bg = bg_per_seq if bg_per_seq.shape == img_rgb01.shape else cv2.resize(bg_per_seq, (img_rgb01.shape[1], img_rgb01.shape[0]))
            res = infer_pair(model, img_rgb01, bg, device=device, tdf_threshold=args.tdf_thr)

            # Match GT
            if f.gt_bbox_yxyx is not None:
                matched = False
                best = None
                for t in res.trajectories:
                    iou = bbox_iou(t.bbox, f.gt_bbox_yxyx)
                    if iou >= 0.1:
                        matched = True
                        if best is None or iou > bbox_iou(best.bbox, f.gt_bbox_yxyx):
                            best = t
                if matched: tp += 1
                else:      fn += 1
                fp += max(0, len(res.trajectories) - (1 if matched else 0))
                # TIoU on best match
                if best is not None and f.gt_traj_yx is not None and f.gt_radius_px is not None:
                    tiou = trajectory_iou(f.gt_traj_yx, best.points,
                                          f.gt_radius_px, img_rgb01.shape[:2])
                    tious.append(tiou)
            else:
                fp += len(res.trajectories)

            # Draw + save every N-th frame
            if f.frame_idx % args.save_every == 0:
                annot = draw_annotated(img_bgr, f.gt_bbox_yxyx, res.trajectories)
                cv2.imwrite(str(seq_dir / f"f{f.frame_idx:04d}.png"), annot)
                if len(saved_grid_imgs) < 4:
                    saved_grid_imgs.append(annot)

        all_tp += tp; all_fp += fp; all_fn += fn
        all_tious.extend(tious)
        p, r = precision_recall(tp, fp, fn)
        per_seq[name] = {
            "n_frames": len(frames), "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r,
            "mean_tiou": float(np.mean(tious)) if tious else None,
        }
        print(f"  {name}: P={p:.3f} R={r:.3f} TIoU={per_seq[name]['mean_tiou']}")

        # Build grid row: 4 sample frames stitched horizontally + sequence label
        if saved_grid_imgs:
            row = np.hstack([cv2.resize(im, (320, 240)) for im in saved_grid_imgs[:4]])
            cv2.putText(row, name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(row, name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 1, cv2.LINE_AA)
            grid_samples.append(row)

    p, r = precision_recall(all_tp, all_fp, all_fn)
    summary = {
        "ckpt": str(args.ckpt),
        "dataset": str(args.dataset),
        "n_sequences": len(seqs),
        "n_frames": sum(len(v) for v in seqs.values()),
        "aggregate": {
            "tp": all_tp, "fp": all_fp, "fn": all_fn,
            "precision": p, "recall": r,
            "mean_tiou": float(np.mean(all_tious)) if all_tious else None,
        },
        "per_sequence": per_seq,
    }
    (args.out / "_metrics.json").write_text(json.dumps(summary, indent=2))
    if grid_samples:
        grid = np.vstack(grid_samples)
        cv2.imwrite(str(args.out / "_grid.png"), grid)

    print("\n=== AGGREGATE ===")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"Mean TIoU: {summary['aggregate']['mean_tiou']}")
    print(f"\nWrote {args.out}/_metrics.json + per-frame overlays + _grid.png")


if __name__ == "__main__":
    main()
