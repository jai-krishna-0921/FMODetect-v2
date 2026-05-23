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
    out_shape: tuple[int, int] = (256, 512)          # H, W
    bg_size_min: int = 256
    speed_range: tuple[float, float] = (0.8, 9.0)    # × disc radius (lower min for slow objects)
    radius_range: tuple[int, int] = (8, 50)          # broadened: pen-thin to ball-size
    aspect_ratio_range: tuple[float, float] = (0.3, 1.0)  # 1.0 = disc, <1 = elongated
    tdf_truncation_factor: float = 2.0               # paper uses 2r

    # --- v2 augmentations & motion variety ---
    augment_prob: float = 0.7                        # apply post-compose augmentations w.p.
    color_jitter_strength: float = 0.2               # ± fraction on brightness/contrast
    gauss_noise_std: float = 0.015                   # std of additive Gaussian noise
    jpeg_prob: float = 0.3                           # apply jpeg-compression simulation
    rotate_fg_prob: float = 0.5                      # rotate FG before convolve
    motion_kinds_weights: tuple[float, ...] = (      # categorical weights for trajectory types
        0.15,  # parabola
        0.20,  # bounce (broken line)
        0.40,  # straight line
        0.10,  # zigzag (3-segment broken line)
        0.10,  # sinusoidal
        0.05,  # accelerating line
    )


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


def _categorical(weights: tuple[float, ...], rng: random.Random) -> int:
    """Pick an index from a weights tuple."""
    total = sum(weights)
    u = rng.random() * total
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if u <= cum:
            return i
    return len(weights) - 1


def _render_polyline(h: int, w: int, pts_yx: np.ndarray) -> np.ndarray:
    """Rasterize an explicit polyline (not parametric) into an AA blur kernel.

    pts_yx: (N, 2) in (y, x) image coords.
    """
    H = np.zeros((h, w), dtype=np.float32)
    for i in range(len(pts_yx) - 1):
        y0, x0 = pts_yx[i]
        y1, x1 = pts_yx[i + 1]
        rr, cc, val = line_aa(int(round(y0)), int(round(x0)),
                              int(round(y1)), int(round(x1)))
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        H[rr[m], cc[m]] = np.maximum(H[rr[m], cc[m]], val[m])
    return H


def _random_trajectory(
    h: int, w: int, rad: int, rng: random.Random, *,
    weights: tuple[float, ...] = (0.15, 0.20, 0.40, 0.10, 0.10, 0.05),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a trajectory.

    Returns (parametric_params (2,4), explicit_polyline (N,2)).
    The explicit polyline is what we actually rasterize for H. The parametric
    `params` is for the parametric-fit head / eval-time supervision (matches paper).
    Zigzag and sinusoidal trajectories can't be represented by the 8-DOF parametric
    form exactly; for them we return the best-fitting line in `params`.
    """
    speed = rng.uniform(0.8, 9.0) * rad
    sy = rng.uniform(0, h - 1)
    sx = rng.uniform(0, w - 1)
    params = np.zeros((2, 4), dtype=np.float32)
    params[:, 0] = (sy, sx)
    kind = _categorical(weights, rng)

    if kind == 0:  # parabola
        ori = rng.uniform(0, 2 * math.pi)
        params[0, 1] = math.sin(ori) * speed
        params[1, 1] = math.cos(ori) * speed
        params[:, 2] = (rng.uniform(10, 20), rng.uniform(10, 20))
        ns = max(2, int(round(speed / 3)))
        ts = np.linspace(0, 1, ns)
        ys = params[0, 0] + params[0, 1] * ts + params[0, 2] * ts * ts
        xs = params[1, 0] + params[1, 1] * ts + params[1, 2] * ts * ts
        return params, np.column_stack([ys, xs])

    if kind == 1:  # bounce (2-segment broken line)
        prc = rng.uniform(0.3, 0.7)
        ori1 = rng.uniform(0, 2 * math.pi)
        while True:
            ori2 = rng.uniform(0, 2 * math.pi)
            d = (ori2 - ori1 + 3 * math.pi) % (2 * math.pi) - math.pi
            if math.pi / 6 < abs(d) < 5 * math.pi / 6:
                break
        params[0, 1] = math.sin(ori1) * speed * prc
        params[1, 1] = math.cos(ori1) * speed * prc
        params[0, 3] = math.sin(ori2) * speed * (1 - prc)
        params[1, 3] = math.cos(ori2) * speed * (1 - prc)
        mid_y = sy + params[0, 1]
        mid_x = sx + params[1, 1]
        end_y = mid_y + params[0, 3]
        end_x = mid_x + params[1, 3]
        return params, np.array([[sy, sx], [mid_y, mid_x], [end_y, end_x]])

    if kind == 3:  # zigzag (3-segment broken line, not exactly parametrizable)
        n_seg = 3
        oris = [rng.uniform(0, 2 * math.pi)]
        for _ in range(n_seg - 1):
            while True:
                o = rng.uniform(0, 2 * math.pi)
                d = (o - oris[-1] + 3 * math.pi) % (2 * math.pi) - math.pi
                if math.pi / 4 < abs(d) < 3 * math.pi / 4:
                    break
            oris.append(o)
        per_seg = speed / n_seg
        cur = np.array([sy, sx], dtype=np.float64)
        pts = [cur.copy()]
        for o in oris:
            cur = cur + np.array([math.sin(o), math.cos(o)]) * per_seg
            pts.append(cur.copy())
        # Best-fit line in params (placeholder; loss isn't using `traj` directly)
        params[:, 1] = (pts[-1] - pts[0])
        return params, np.array(pts)

    if kind == 4:  # sinusoidal
        ori = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(rad * 0.5, rad * 2.0)
        freq = rng.uniform(1.0, 3.0)
        n = max(3, int(round(speed / 2)))
        ts = np.linspace(0, 1, n)
        base_y = sy + math.sin(ori) * speed * ts
        base_x = sx + math.cos(ori) * speed * ts
        perp_y = -math.cos(ori) * np.sin(2 * math.pi * freq * ts) * amp
        perp_x =  math.sin(ori) * np.sin(2 * math.pi * freq * ts) * amp
        ys = base_y + perp_y
        xs = base_x + perp_x
        params[0, 1] = math.sin(ori) * speed
        params[1, 1] = math.cos(ori) * speed
        return params, np.column_stack([ys, xs])

    if kind == 5:  # accelerating line
        ori = rng.uniform(0, 2 * math.pi)
        # Acceleration along the same direction (so c1 + c2 -> longer)
        accel = rng.uniform(0.4, 0.9)
        params[0, 1] = math.sin(ori) * speed * (1 - accel)
        params[1, 1] = math.cos(ori) * speed * (1 - accel)
        params[0, 2] = math.sin(ori) * speed * accel
        params[1, 2] = math.cos(ori) * speed * accel
        ns = max(2, int(round(speed / 3)))
        ts = np.linspace(0, 1, ns)
        ys = params[0, 0] + params[0, 1] * ts + params[0, 2] * ts * ts
        xs = params[1, 0] + params[1, 1] * ts + params[1, 2] * ts * ts
        return params, np.column_stack([ys, xs])

    # default: straight line
    ori = rng.uniform(0, 2 * math.pi)
    params[0, 1] = math.sin(ori) * speed
    params[1, 1] = math.cos(ori) * speed
    end_y = sy + params[0, 1]
    end_x = sx + params[1, 1]
    return params, np.array([[sy, sx], [end_y, end_x]])


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


def _augment(img: np.ndarray, bgr: np.ndarray, cfg: SynthConfig,
             rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation to the *composited image* (and matching bgr for consistency).

    Augmentations applied independently to img and bgr would break the formation
    invariant we want the network to learn, so we couple them: identical noise
    seed, same brightness/contrast jitter, same JPEG simulation. Order: brightness,
    contrast, noise, jpeg.
    """
    if rng.random() > cfg.augment_prob:
        return img, bgr

    # Same scalar jitter applied to both
    s = cfg.color_jitter_strength
    b = 1.0 + rng.uniform(-s, s)   # brightness multiplier
    c = 1.0 + rng.uniform(-s, s)   # contrast multiplier
    mean = 0.5

    def apply(x: np.ndarray) -> np.ndarray:
        x = x * b
        x = (x - mean) * c + mean
        if cfg.gauss_noise_std > 0:
            x = x + np.random.normal(0, cfg.gauss_noise_std, x.shape).astype(np.float32)
        return np.clip(x, 0.0, 1.0)

    img2 = apply(img)
    bgr2 = apply(bgr)

    # JPEG simulation (encode + decode once with PIL)
    if rng.random() < cfg.jpeg_prob:
        from io import BytesIO
        q = rng.randint(45, 90)
        def jpeg(x: np.ndarray) -> np.ndarray:
            pil = Image.fromarray((x * 255).astype(np.uint8))
            buf = BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return np.asarray(Image.open(buf)).astype(np.float32) / 255.0
        img2 = jpeg(img2)
        bgr2 = jpeg(bgr2)

    return img2, bgr2


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

    # Resize pattern with random short/long axis (aspect ratio jitter for elongated FMOs).
    rad = rng.randint(*cfg.radius_range)
    short = 2 * rad
    ar = rng.uniform(*cfg.aspect_ratio_range)        # 1.0 = circular disc, <1 = elongated
    long_axis = max(short, int(round(short / ar)))
    # randomize which dimension is the long axis (vertical vs horizontal elongation)
    if rng.random() < 0.5:
        w, h = long_axis, short
    else:
        w, h = short, long_axis
    pat = np.asarray(Image.fromarray(pattern_rgba).resize((w, h), Image.BILINEAR))
    # Optional rotation of foreground before convolving (varies texture orientation)
    if rng.random() < cfg.rotate_fg_prob:
        deg = rng.uniform(0, 360)
        pat = np.asarray(Image.fromarray(pat).rotate(deg, resample=Image.BILINEAR))
    F = (pat[..., :3].astype(np.float32) / 255.0)
    Mdisc = (pat[..., 3].astype(np.float32) / 255.0)
    FM = F * Mdisc[..., None]

    # Sample trajectory
    traj_params, polyline = _random_trajectory(H, W, rad, rng,
                                                weights=cfg.motion_kinds_weights)
    h_kernel = _render_polyline(H, W, polyline)
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

    # Augmentation
    img, bg_aug = _augment(img, bg_resized, cfg, rng)

    # TDF target
    tdf = _truncated_distance_function(h_kernel, rad, trunc_factor=cfg.tdf_truncation_factor)

    return {
        "image": img,
        "bgr": bg_aug,
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
