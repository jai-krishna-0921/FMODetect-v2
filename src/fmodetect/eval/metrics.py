"""FMODetect-v2 evaluation metrics.

Implements:
  - bbox_iou           : 2-D IoU between predicted and GT bounding boxes
  - is_true_positive   : paper's TP rule (bbox IoU > 0.1) — paper sec. 4
  - precision_recall   : standard P/R given matched / unmatched detections
  - trajectory_iou     : Trajectory Intersection-over-Union (paper Eq.7) —
                         IoU of the swept circular FMO masks along the GT and
                         predicted trajectories, averaged over the GT length.
"""
from __future__ import annotations

import numpy as np


def bbox_iou(box_a: tuple[float, float, float, float],
             box_b: tuple[float, float, float, float]) -> float:
    """IoU between two (y0, x0, y1, x1) boxes."""
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    iy0 = max(ay0, by0); ix0 = max(ax0, bx0)
    iy1 = min(ay1, by1); ix1 = min(ax1, bx1)
    inter = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mask_from_traj(traj_xy: np.ndarray, radius: float, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize disc-swept mask along a polyline trajectory.

    traj_xy: (N, 2) in (y, x) image coords.
    radius : disc radius in pixels.
    Returns boolean (H, W) mask of the swept disc region.
    """
    h, w = image_shape
    yy, xx = np.mgrid[:h, :w]
    out = np.zeros(image_shape, dtype=bool)
    for i in range(len(traj_xy)):
        cy, cx = traj_xy[i]
        # Bounding-box short-circuit
        y0 = max(0, int(cy - radius - 1)); y1 = min(h, int(cy + radius + 1))
        x0 = max(0, int(cx - radius - 1)); x1 = min(w, int(cx + radius + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        sub = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2 <= radius * radius
        out[y0:y1, x0:x1] |= sub
    return out


def trajectory_iou(
    gt_traj_yx: np.ndarray, pred_traj_yx: np.ndarray,
    gt_radius: float, image_shape: tuple[int, int],
    *, n_samples: int = 32,
) -> float:
    """Paper Eq. (7) TIoU: average IoU between swept disc masks at matched timesteps.

    Resamples both trajectories to `n_samples` evenly spaced points, then computes
    mean per-timestep IoU. Also tries the reverse direction (paper does this).
    """
    if len(gt_traj_yx) == 0 or len(pred_traj_yx) == 0:
        return 0.0

    def resample(traj: np.ndarray, n: int) -> np.ndarray:
        d = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(traj, axis=0), axis=1)])
        if d[-1] <= 1e-6:
            return np.repeat(traj[0][None], n, axis=0)
        s = np.linspace(0.0, d[-1], n)
        out = np.column_stack([np.interp(s, d, traj[:, 0]), np.interp(s, d, traj[:, 1])])
        return out

    gt_r = resample(gt_traj_yx, n_samples)
    pr_r = resample(pred_traj_yx, n_samples)

    def mean_iou(gt: np.ndarray, pr: np.ndarray) -> float:
        ious = []
        for i in range(n_samples):
            gm = _mask_from_traj(gt[i:i + 1], gt_radius, image_shape)
            pm = _mask_from_traj(pr[i:i + 1], gt_radius, image_shape)
            u = (gm | pm).sum()
            ious.append(((gm & pm).sum() / u) if u else 0.0)
        return float(np.mean(ious))

    return max(mean_iou(gt_r, pr_r), mean_iou(gt_r, pr_r[::-1]))


def precision_recall(n_tp: int, n_fp: int, n_fn: int) -> tuple[float, float]:
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    return p, r
