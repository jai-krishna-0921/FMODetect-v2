"""Procedural geometric pattern generator for synthetic FMO foregrounds.

Replaces the original repo's dependency on the (PyPI-broken) `geopatterns` +
`cairosvg` SVG pipeline. Generates 100 deterministic textured disc patterns
using only numpy + pillow — no SVG, no Cairo.

Patterns are: stripes, checkerboard, concentric rings, dots grid, plaid,
chevrons, triangles, hexagons. All masked to a disc.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PATTERN_KINDS = (
    "stripes", "checker", "rings", "dots", "plaid", "chevrons", "triangles", "hex",
)


def _random_palette(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Pick two contrasting RGB colors."""
    def rand_color() -> tuple[int, int, int]:
        return (rng.randint(20, 235), rng.randint(20, 235), rng.randint(20, 235))
    c1 = rand_color()
    c2 = rand_color()
    # Force contrast: if too close, perturb c2
    if sum(abs(a - b) for a, b in zip(c1, c2)) < 120:
        c2 = tuple(255 - x for x in c1)  # type: ignore[assignment]
    return c1, c2


def _draw_pattern(kind: str, size: int, rng: random.Random) -> Image.Image:
    c1, c2 = _random_palette(rng)
    img = Image.new("RGB", (size, size), c1)
    drw = ImageDraw.Draw(img)
    period = rng.randint(8, max(9, size // 4))

    if kind == "stripes":
        for x in range(0, size, period):
            drw.rectangle((x, 0, x + period // 2, size), fill=c2)
    elif kind == "checker":
        for y in range(0, size, period):
            for x in range(0, size, period):
                if ((x // period) + (y // period)) & 1:
                    drw.rectangle((x, y, x + period, y + period), fill=c2)
    elif kind == "rings":
        cx, cy = size // 2, size // 2
        for r in range(period, size, period):
            drw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=c2, width=max(2, period // 4))
    elif kind == "dots":
        r = max(3, period // 3)
        for y in range(period // 2, size, period):
            for x in range(period // 2, size, period):
                drw.ellipse((x - r, y - r, x + r, y + r), fill=c2)
    elif kind == "plaid":
        for x in range(0, size, period):
            drw.rectangle((x, 0, x + period // 2, size), fill=c2)
        for y in range(0, size, period):
            drw.rectangle((0, y, size, y + period // 2), fill=c2)
    elif kind == "chevrons":
        for y in range(0, size, period):
            pts = []
            for x in range(0, size + period, period):
                pts.append((x, y + (period // 2 if (x // period) & 1 else 0)))
            drw.line(pts, fill=c2, width=max(2, period // 4))
    elif kind == "triangles":
        for y in range(0, size, period):
            for x in range(0, size, period):
                up = ((x // period) + (y // period)) & 1
                if up:
                    drw.polygon([(x, y + period), (x + period, y + period), (x + period // 2, y)], fill=c2)
                else:
                    drw.polygon([(x, y), (x + period, y), (x + period // 2, y + period)], fill=c2)
    elif kind == "hex":
        r = period
        h = int(r * math.sqrt(3))
        for row, y in enumerate(range(-r, size + r, h)):
            off = r * 1.5 if row & 1 else 0
            for x in range(-r, size + 2 * r, 3 * r):
                cx = x + off
                pts = [(cx + r * math.cos(math.pi / 3 * i), y + r * math.sin(math.pi / 3 * i)) for i in range(6)]
                drw.polygon(pts, outline=c2, width=2)
    else:
        raise ValueError(kind)
    return img


def make_disc_pattern(size: int, *, rng: random.Random) -> np.ndarray:
    """Return a (size, size, 4) uint8 RGBA array with a textured disc."""
    kind = rng.choice(PATTERN_KINDS)
    pat = _draw_pattern(kind, size, rng).convert("RGBA")
    # Disc alpha mask
    alpha = Image.new("L", (size, size), 0)
    ImageDraw.Draw(alpha).ellipse((0, 0, size - 1, size - 1), fill=255)
    pat.putalpha(alpha)
    return np.asarray(pat)


def generate_pattern_bank(out_dir: Path, count: int = 100, seed: int = 1234) -> list[Path]:
    """Pre-generate `count` disc patterns of random sizes and save as PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for i in range(count):
        size = rng.randint(40, 120)  # disc diameter in px
        arr = make_disc_pattern(size, rng=rng)
        p = out_dir / f"pattern_{i:04d}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/patterns"))
    ap.add_argument("--count", type=int, default=100)
    args = ap.parse_args()
    paths = generate_pattern_bank(args.out, args.count)
    print(f"Wrote {len(paths)} patterns to {args.out}")
