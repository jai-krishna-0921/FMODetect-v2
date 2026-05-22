"""Synthetic FMO sample generator.

Implements the paper's image formation model
    I = H * F + (1 - H * M) * B
where
    H : 2D trajectory blur kernel (delta line/parabola, normalized to sum=1)
    F : sharp object appearance (textured disc), centered, with mask M
    M : binary/soft mask of F
    B : background (3-frame median to approximate clean bg)

Outputs per sample, all numpy arrays:
    image  (H, W, 3) float32 in [0, 1]
    bgr    (H, W, 3) float32 in [0, 1]
    tdf    (H, W)    float32 in [0, 1]   — 1 = on trajectory, 0 = far
    hm     (H, W)    float32 in [0, 1]   — blurred mask H*M (matting target)
    traj   (2, 4)    float32              — parametric trajectory (8 DOF)

This module produces *one sample at a time*. A WebDataset writer wraps it for
disk caching (see `src/fmodetect/data/build_dataset.py`).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, signal
from skimage.draw import line_aa
from skimage.transform import resize


@dataclass
class SynthConfig:
    out_shape: tuple[int, int] = (256, 512)         # H, W
    bg_size_min: int = 256
    speed_range: tuple[float, float] = (1.5, 9.0)    # × disc radius
    radius_range: tuple[int, int] = (20, 60)         # disc radius in px (smaller than paper to fit 4GB)
    tdf_truncation_factor: float = 2.0               # paper uses 2r


def _disc_mask(rad: int) -> np.ndarray:
    sz = 2 * rad
    yy, xx = np.mgrid[:sz, :sz]
    cy = cx = (sz - 1) / 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad).astype(np.float32)


def _render_trajectory(h: int, w: int, params: np.ndarray) -> np.ndarray:
    """Rasterize a parametric trajectory onto an (H, W) image with anti-aliasing.

    params: (2, 4) — rows are y/x, cols are c0,c1,c2,c3 such that
        C(t) = c0 + c1 * min(2t,1) + c2 * min(2t,1)^2 + c3 * max(2t-1, 0),  t in [0,1].
    """
    H = np.zeros((h, w), dtype=np.float32)
    ns = max(2, int(round(np.linalg.norm(params[:, 1]) / 3)))
    ts = np.linspace(0.0, 1.0, ns)
    a = np.minimum(2 * ts, 1.0)
    b = np.maximum(2 * ts - 1.0, 0.0)
    ys = params[0, 0] + params[0, 1] * a + params[0, 2] * a * a + params[0, 3] * b
    xs = params[1, 0] + params[1, 1] * a + params[1, 2] * a * a + params[1, 3] * b
    for i in range(ns - 1):
        rr, cc, val = line_aa(int(round(ys[i])), int(round(xs[i])),
                              int(round(ys[i + 1])), int(round(xs[i + 1])))
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        H[rr[m], cc[m]] = np.maximum(H[rr[m], cc[m]], val[m])
    return H


def _random_trajectory(h: int, w: int, rad: int, rng: random.Random) -> np.ndarray:
    """Sample one of: pure line, parabola, broken line. Returns (2, 4) params."""
    speed = rng.uniform(1.5, 9.0) * rad
    sy = rng.randint(0, h - 1)
    sx = rng.randint(0, w - 1)
    params = np.zeros((2, 4), dtype=np.float32)
    params[:, 0] = (sy, sx)
    kind = rng.randint(0, 9)
    if kind == 0:  # parabola
        ori = rng.uniform(0, 2 * math.pi)
        params[0, 1] = math.sin(ori) * speed
        params[1, 1] = math.cos(ori) * speed
        params[:, 2] = (rng.uniform(10, 20), rng.uniform(10, 20))
    elif kind == 1:  # broken line (bounce)
        prc = rng.uniform(0.3, 0.7)
        ori1 = rng.uniform(0, 2 * math.pi)
        # second leg with > 30° turn
        while True:
            ori2 = rng.uniform(0, 2 * math.pi)
            d = (ori2 - ori1 + 3 * math.pi) % (2 * math.pi) - math.pi
            if math.pi / 6 < abs(d) < 5 * math.pi / 6:
                break
        params[0, 1] = math.sin(ori1) * speed * prc
        params[1, 1] = math.cos(ori1) * speed * prc
        params[0, 3] = math.sin(ori2) * speed * (1 - prc)
        params[1, 3] = math.cos(ori2) * speed * (1 - prc)
    else:  # straight line
        ori = rng.uniform(0, 2 * math.pi)
        params[0, 1] = math.sin(ori) * speed
        params[1, 1] = math.cos(ori) * speed
    return params


def _truncated_distance_function(traj: np.ndarray, rad: int, *, trunc_factor: float = 2.0) -> np.ndarray:
    """TDF as defined in paper Eq. (2): 1 - min(1, ||x - C(t)||/(trunc_factor*r))."""
    bin_img = (traj > 1e-3).astype(np.uint8)
    if bin_img.sum() == 0:
        return np.zeros_like(traj)
    dt = ndimage.distance_transform_edt(1 - bin_img)
    tdf = 1.0 - np.minimum(1.0, dt / (trunc_factor * rad))
    return tdf.astype(np.float32)


def _normalize_kernel(h: np.ndarray) -> np.ndarray:
    s = h.sum()
    return h / s if s > 0 else h


def synthesize_sample(
    bg: np.ndarray,
    pattern_rgba: np.ndarray,
    cfg: SynthConfig,
    rng: random.Random,
) -> dict[str, np.ndarray] | None:
    """Compose one FMO sample. Returns dict or None if the sample is degenerate."""
    H, W = cfg.out_shape

    # Resize bg
    bg_resized = resize(bg, (H, W), order=3, anti_aliasing=True).astype(np.float32)

    # Resize pattern to a random disc diameter
    rad = rng.randint(*cfg.radius_range)
    diam = 2 * rad
    pat = np.asarray(Image.fromarray(pattern_rgba).resize((diam, diam), Image.BILINEAR))
    F = (pat[..., :3].astype(np.float32) / 255.0)
    Mdisc = (pat[..., 3].astype(np.float32) / 255.0)
    FM = F * Mdisc[..., None]

    # Sample trajectory and rasterize a thin (delta-line) blur kernel
    traj_params = _random_trajectory(H, W, rad, rng)
    h_kernel = _render_trajectory(H, W, traj_params)
    if h_kernel.sum() < 5:
        return None
    h_norm = _normalize_kernel(h_kernel)

    # Convolve: HM (blurred mask) and HF (blurred appearance)
    hm = signal.fftconvolve(h_norm, Mdisc, mode="same").astype(np.float32)
    hm = np.clip(hm, 0.0, 1.0)
    hf = np.stack(
        [signal.fftconvolve(h_norm, FM[..., c], mode="same") for c in range(3)],
        axis=-1,
    ).astype(np.float32)
    hf = np.clip(hf, 0.0, 1.0)

    # Image formation
    img = bg_resized * (1.0 - hm[..., None]) + hf
    img = np.clip(img, 0.0, 1.0)

    # TDF target
    tdf = _truncated_distance_function(h_kernel, rad, trunc_factor=cfg.tdf_truncation_factor)

    return {
        "image": img,
        "bgr": bg_resized,
        "tdf": tdf,
        "hm": hm,
        "traj": traj_params,
        "rad": np.float32(rad),
    }


def load_background_paths(vot_root: Path) -> list[Path]:
    """List every JPEG frame in a VOT2016 sequence tree."""
    return sorted(Path(vot_root).rglob("*.jpg"))


def load_pattern_paths(pat_root: Path) -> list[Path]:
    return sorted(Path(pat_root).glob("*.png"))


def read_image(p: Path) -> np.ndarray:
    """Read RGB image as float32 [0,1]."""
    img = np.asarray(Image.open(p).convert("RGB"))
    return img.astype(np.float32) / 255.0


def read_rgba(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGBA"))
