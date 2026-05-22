"""Smoke tests for the model + losses. Requires torch."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.fmodetect.models.losses import combined_loss, tdf_l1
from src.fmodetect.models.unet import FMODetectNet, UNetConfig


def test_forward_shapes_cpu() -> None:
    net = FMODetectNet(UNetConfig()).eval()
    x = torch.randn(1, 6, 64, 128)  # small for CPU
    out = net(x)
    assert out["tdf"].shape == (1, 1, 64, 128)
    assert out["log_var_tdf"].shape == (1, 1, 64, 128)
    assert out["hm"].shape == (1, 1, 64, 128)
    # tdf and hm should be in [0, 1] after sigmoid
    assert (out["tdf"] >= 0).all() and (out["tdf"] <= 1).all()
    assert (out["hm"] >= 0).all() and (out["hm"] <= 1).all()


def test_param_count_under_budget() -> None:
    """Params should comfortably fit on a 4 GB GPU."""
    net = FMODetectNet(UNetConfig())
    n = sum(p.numel() for p in net.parameters())
    # Paper detection net is 4.8M; matting net adds another decoder.
    # We expect ~5-8M total; assert generous upper bound.
    assert n < 10e6, f"too many params: {n/1e6:.2f}M"


def test_loss_paper_baseline() -> None:
    pred = torch.rand(2, 1, 16, 16)
    gt = torch.rand(2, 1, 16, 16)
    gt = (gt > 0.5).float() * gt  # make some zeros
    loss = tdf_l1(pred, gt)
    assert loss.item() >= 0


def test_combined_loss_runs() -> None:
    net = FMODetectNet(UNetConfig())
    x = torch.randn(1, 6, 64, 128)
    out = net(x)
    tgt = {"tdf": torch.rand(1, 1, 64, 128), "hm": torch.rand(1, 1, 64, 128)}
    parts = combined_loss(out, tgt)
    assert "total" in parts
    parts["total"].backward()
    # Make sure at least one param has a grad
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients computed"
