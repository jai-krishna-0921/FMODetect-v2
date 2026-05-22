"""Smoke tests for the synthetic data pipeline. Does not require torch."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.fmodetect.data.patterns import generate_pattern_bank, make_disc_pattern
from src.fmodetect.data.synthesize import SynthConfig, synthesize_sample


def test_make_disc_pattern_shape() -> None:
    rng = random.Random(0)
    arr = make_disc_pattern(64, rng=rng)
    assert arr.shape == (64, 64, 4)
    # Alpha mask should be ~disc area
    alpha = arr[..., 3]
    area = (alpha > 0).sum()
    expected = np.pi * (64 / 2) ** 2
    assert 0.85 * expected < area < 1.05 * expected


def test_synthesize_sample_outputs() -> None:
    rng = random.Random(0)
    cfg = SynthConfig(out_shape=(128, 256), radius_range=(15, 30))
    bg = np.random.rand(200, 400, 3).astype(np.float32)
    pat = make_disc_pattern(64, rng=rng)
    # Try a few times — some configs are rejected
    for _ in range(10):
        s = synthesize_sample(bg, pat, cfg, rng)
        if s is not None:
            break
    assert s is not None
    assert s["image"].shape == (128, 256, 3)
    assert s["bgr"].shape == (128, 256, 3)
    assert s["tdf"].shape == (128, 256)
    assert s["hm"].shape == (128, 256)
    assert s["traj"].shape == (2, 4)
    assert 0.0 <= s["image"].min() and s["image"].max() <= 1.0
    assert 0.0 <= s["tdf"].min() and s["tdf"].max() <= 1.0
    # Trajectory must have at least some "on-trajectory" pixels (tdf == 1)
    assert (s["tdf"] >= 0.99).any()


def test_pattern_bank(tmp_path) -> None:
    paths = generate_pattern_bank(tmp_path, count=8, seed=1)
    assert len(paths) == 8
    assert all(p.exists() for p in paths)
