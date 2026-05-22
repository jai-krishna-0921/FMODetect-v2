"""Build a synthetic VOT-FMO dataset on disk as an H5 file (paper-faithful).

Usage:
    .venv/bin/python -m src.fmodetect.data.build_dataset \\
        --bg datasets/vot2016 \\
        --patterns datasets/patterns \\
        --out datasets/synth/vot_fmo.h5 \\
        --n 5000

Output H5 layout:
    /sample_00000000/image   float32 (H, W, 3)
    /sample_00000000/bgr     float32 (H, W, 3)
    /sample_00000000/tdf     float32 (H, W)
    /sample_00000000/hm      float32 (H, W)
    /sample_00000000/traj    float32 (2, 4)
    /sample_00000000/rad     float32 ()
    ...
Attributes on root: 'n_samples', 'out_shape', 'seed'.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .patterns import generate_pattern_bank
from .synthesize import (
    SynthConfig,
    load_background_paths,
    load_pattern_paths,
    read_image,
    read_rgba,
    synthesize_sample,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bg", type=Path, required=True, help="VOT2016 root (recursive .jpg)")
    p.add_argument("--patterns", type=Path, required=True, help="Pattern bank dir (.png)")
    p.add_argument("--out", type=Path, required=True, help="Output H5 path")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--shape", type=int, nargs=2, default=(256, 512), help="H W")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patterns-if-empty", type=int, default=100,
                   help="Auto-generate this many patterns if --patterns is empty.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    cfg = SynthConfig(out_shape=tuple(args.shape))

    # Patterns
    pat_paths = load_pattern_paths(args.patterns)
    if not pat_paths:
        print(f"[build_dataset] {args.patterns} empty — generating {args.patterns_if_empty} patterns")
        generate_pattern_bank(args.patterns, args.patterns_if_empty, seed=args.seed)
        pat_paths = load_pattern_paths(args.patterns)
    assert pat_paths, "no patterns even after auto-generation"

    # Backgrounds
    bg_paths = load_background_paths(args.bg)
    assert bg_paths, f"no .jpg found under {args.bg}"
    print(f"[build_dataset] {len(bg_paths)} bg frames, {len(pat_paths)} patterns")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    written = 0
    rejected = 0
    with h5py.File(args.out, "w") as f:
        f.attrs["n_samples_target"] = args.n
        f.attrs["out_shape"] = list(args.shape)
        f.attrs["seed"] = args.seed
        with tqdm(total=args.n, desc="synth") as bar:
            while written < args.n:
                bg_p = pat_paths[rng.randrange(len(pat_paths))]  # placeholder, see below
                bg_path = bg_paths[rng.randrange(len(bg_paths))]
                pat_path = pat_paths[rng.randrange(len(pat_paths))]
                bg = read_image(bg_path)
                pat = read_rgba(pat_path)
                sample = synthesize_sample(bg, pat, cfg, rng)
                if sample is None:
                    rejected += 1
                    continue
                g = f.create_group(f"sample_{written:08d}")
                for k, v in sample.items():
                    kwargs = {}
                    if hasattr(v, "shape") and v.shape:  # h5py: no compression on scalars
                        kwargs["compression"] = "lzf"
                    g.create_dataset(k, data=v, **kwargs)
                written += 1
                bar.update(1)
                if rejected and written % 500 == 0:
                    bar.set_postfix(rejected=rejected)
        f.attrs["n_samples"] = written

    print(f"[build_dataset] wrote {written} samples to {args.out} ({rejected} rejected)")


if __name__ == "__main__":
    main()
