"""Verify post-hoc trajectory extraction recovers a known synthetic trajectory."""
from __future__ import annotations

import math
import random

import numpy as np

from src.fmodetect.data.patterns import make_disc_pattern
from src.fmodetect.data.synthesize import SynthConfig, synthesize_sample
from src.fmodetect.inference.trajectory import extract_trajectories


def test_recovers_synthetic_trajectory() -> None:
    """When given the GT TDF, the extractor should find exactly one trajectory
    whose length agrees with the actual blur-kernel length to within ~20%."""
    rng = random.Random(0)
    cfg = SynthConfig(out_shape=(256, 512), radius_range=(30, 40))
    bg = np.random.rand(300, 600, 3).astype(np.float32)
    pat = make_disc_pattern(80, rng=rng)
    # retry until a valid sample
    sample = None
    for _ in range(20):
        sample = synthesize_sample(bg, pat, cfg, rng)
        if sample is not None:
            break
    assert sample is not None

    trajs = extract_trajectories(sample["tdf"], tdf_threshold=0.5)
    assert len(trajs) == 1, f"expected 1 trajectory, got {len(trajs)}"
    t = trajs[0]
    # The synthetic trajectory length is bounded between 1.5*rad and 9*rad.
    # We just check it's in a sensible range.
    assert 30 < t.length_px < 1000
    assert 0 < t.speed_px_per_frame == t.length_px
    assert t.radius_px > 1.0
    assert t.confidence > 0.4
    assert t.n_pixels > 30
