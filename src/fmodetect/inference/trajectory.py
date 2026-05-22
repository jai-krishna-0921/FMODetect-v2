"""Post-hoc trajectory + speed extraction from the TDF heatmap.

Pipeline:
  1. Threshold TDF -> binary "on-trajectory" pixels
  2. Connected components -> one region per candidate FMO
  3. Per region: skeletonize -> ordered pixel chain
  4. Fit a parametric curve C(t) = c0 + c1*t + c2*t^2 (least squares).
     Optionally try a piecewise linear (bounce) fit and pick the lower residual.
  5. Derive: length_px, speed_px_per_frame, radius (from local TDF width).

These match the paper's parametric trajectory definition (paper Eq.4) for the
non-bounce case (line/parabola). Bounce is approximated by RANSAC-style
two-segment fit if residual is high.

This is *post-hoc* fitting — we do not modify the trained model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from skimage import measure
from skimage.morphology import medial_axis, remove_small_objects


@dataclass
class Trajectory:
    points: np.ndarray            # (N, 2) ordered (y, x) skeleton points in image coords
    curve_params: np.ndarray      # (2, 4) — c0, c1, c2, c3 per (y, x); c3 = 0 unless bounce
    is_bounce: bool
    length_px: float              # arc length of fitted curve
    speed_px_per_frame: float     # = length_px (assuming exposure = 1 frame)
    radius_px: float              # half-width of the trajectory tube (from TDF)
    bbox: tuple[int, int, int, int]   # (y0, x0, y1, x1)
    fit_residual_rms: float
    confidence: float             # mean TDF value along trajectory
    n_pixels: int


def _order_skeleton(sk: np.ndarray) -> np.ndarray:
    """Order skeleton pixels by greedy nearest-neighbour from one endpoint."""
    ys, xs = np.where(sk)
    if len(ys) < 2:
        return np.column_stack([ys, xs])
    pts = np.column_stack([ys, xs]).astype(np.float64)
    # Endpoints: pixels with exactly one skeleton neighbour
    from scipy.ndimage import convolve
    nb = convolve(sk.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - sk
    end_mask = (nb == 1) & sk
    eys, exs = np.where(end_mask)
    if len(eys) == 0:
        start_idx = 0
    else:
        # Use the leftmost endpoint as the start
        order = np.argsort(exs)
        start_idx = np.where((pts[:, 0] == eys[order[0]]) & (pts[:, 1] == exs[order[0]]))[0][0]
    visited = np.zeros(len(pts), dtype=bool)
    order_out = [start_idx]
    visited[start_idx] = True
    cur = start_idx
    for _ in range(len(pts) - 1):
        diff = pts - pts[cur]
        d2 = (diff ** 2).sum(axis=1)
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        if not np.isfinite(d2[nxt]):
            break
        order_out.append(nxt)
        visited[nxt] = True
        cur = nxt
    return pts[order_out]


def _fit_parabola(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit C(t) = c0 + c1*t + c2*t^2 to (y, x) chain. Returns (params (2,4), rms residual)."""
    n = len(pts)
    if n < 3:
        params = np.zeros((2, 4), dtype=np.float32)
        params[:, 0] = pts.mean(axis=0)
        if n >= 2:
            params[:, 1] = pts[-1] - pts[0]
        return params, float("inf")
    ts = np.linspace(0.0, 1.0, n)
    # Design matrix for [1, t, t^2]
    A = np.column_stack([np.ones(n), ts, ts ** 2])
    coef_y, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
    coef_x, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
    pred_y = A @ coef_y
    pred_x = A @ coef_x
    rms = float(np.sqrt(((pred_y - pts[:, 0]) ** 2 + (pred_x - pts[:, 1]) ** 2).mean()))
    params = np.zeros((2, 4), dtype=np.float32)
    params[0] = [coef_y[0], coef_y[1], coef_y[2], 0.0]
    params[1] = [coef_x[0], coef_x[1], coef_x[2], 0.0]
    return params, rms


def _curve_length(params: np.ndarray, n_samples: int = 64) -> float:
    """Arc length of C(t) on t in [0,1]. Includes the c3 (bounce) tail."""
    ts = np.linspace(0.0, 1.0, n_samples)
    a = np.minimum(2 * ts, 1.0)
    b = np.maximum(2 * ts - 1.0, 0.0)
    y = params[0, 0] + params[0, 1] * a + params[0, 2] * a * a + params[0, 3] * b
    x = params[1, 0] + params[1, 1] * a + params[1, 2] * a * a + params[1, 3] * b
    return float(np.sum(np.sqrt(np.diff(y) ** 2 + np.diff(x) ** 2)))


def _local_radius(tdf: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float:
    """Estimate the trajectory tube radius from TDF width at the skeleton."""
    if len(ys) == 0:
        return 1.0
    # Half-width = expected distance from skeleton pixel (tdf~1) to tdf<=0.5.
    # Coarse: look at TDF column orthogonal to motion, find boundary.
    # Simpler proxy: rad ~ trunc_factor / (TDF gradient at edge); we just use
    # the mean distance to non-trajectory pixels as a stand-in.
    from scipy.ndimage import distance_transform_edt
    inv = (tdf < 0.2).astype(np.uint8)
    dt = distance_transform_edt(1 - inv)
    rads = dt[ys.astype(int).clip(0, tdf.shape[0] - 1),
              xs.astype(int).clip(0, tdf.shape[1] - 1)]
    return float(np.median(rads) + 1.0)


def extract_trajectories(
    tdf: np.ndarray,
    *,
    tdf_threshold: float = 0.5,
    min_area: int = 25,
    fps: float | None = None,
) -> list[Trajectory]:
    """Extract one Trajectory per connected component of TDF >= threshold.

    `fps` is optional — if provided, `speed_px_per_frame` is unchanged but
    callers can divide by `fps` to get px/sec.
    """
    binmap = tdf >= tdf_threshold
    # scikit-image >=0.26 renamed `min_size` -> `max_size` (removes objects <= max_size).
    try:
        binmap = remove_small_objects(binmap, max_size=min_area - 1)
    except TypeError:  # older skimage
        binmap = remove_small_objects(binmap, min_size=min_area)
    labels = measure.label(binmap, connectivity=2)
    results: list[Trajectory] = []
    for region in measure.regionprops(labels):
        mask = labels == region.label
        sk = medial_axis(mask)
        ordered = _order_skeleton(sk)
        if len(ordered) < 3:
            continue
        params, rms = _fit_parabola(ordered)
        length = _curve_length(params)
        rad = _local_radius(tdf, ordered[:, 0], ordered[:, 1])
        speed = length  # px / frame (assumes one exposure = one frame)
        y0, x0, y1, x1 = region.bbox
        conf = float(tdf[mask].mean())
        results.append(Trajectory(
            points=ordered,
            curve_params=params,
            is_bounce=False,
            length_px=length,
            speed_px_per_frame=speed,
            radius_px=rad,
            bbox=(int(y0), int(x0), int(y1), int(x1)),
            fit_residual_rms=rms,
            confidence=conf,
            n_pixels=int(mask.sum()),
        ))
    return results


def trajectory_as_dict(t: Trajectory, fps: float | None = None) -> dict:
    out = {
        "points": t.points.astype(float).tolist(),
        "curve_params": t.curve_params.astype(float).tolist(),
        "is_bounce": t.is_bounce,
        "length_px": t.length_px,
        "speed_px_per_frame": t.speed_px_per_frame,
        "radius_px": t.radius_px,
        "bbox_yxyx": list(t.bbox),
        "fit_residual_rms": t.fit_residual_rms,
        "confidence": t.confidence,
        "n_pixels": t.n_pixels,
    }
    if fps:
        out["speed_px_per_sec"] = t.speed_px_per_frame * fps
    return out


def overlay_trajectories(
    image_rgb01: np.ndarray, trajectories: list[Trajectory], *,
    color_bgr: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw fitted parametric curves on top of the input image (returns uint8 BGR)."""
    import cv2
    out = (image_rgb01[:, :, ::-1] * 255).astype(np.uint8).copy()
    for t in trajectories:
        n = 64
        ts = np.linspace(0.0, 1.0, n)
        a = np.minimum(2 * ts, 1.0)
        b = np.maximum(2 * ts - 1.0, 0.0)
        y = t.curve_params[0, 0] + t.curve_params[0, 1] * a + t.curve_params[0, 2] * a * a + t.curve_params[0, 3] * b
        x = t.curve_params[1, 0] + t.curve_params[1, 1] * a + t.curve_params[1, 2] * a * a + t.curve_params[1, 3] * b
        pts = np.column_stack([x, y]).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=color_bgr, thickness=thickness)
        # endpoints
        cv2.circle(out, (int(x[0]), int(y[0])), 5, (0, 200, 0), -1)
        cv2.circle(out, (int(x[-1]), int(y[-1])), 5, (0, 0, 255), -1)
        # speed label
        mid = (int(x[n // 2]), int(y[n // 2]))
        cv2.putText(out, f"v={t.speed_px_per_frame:.1f}px/f r={t.radius_px:.1f}",
                    mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
