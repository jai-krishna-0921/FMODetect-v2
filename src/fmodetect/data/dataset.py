"""PyTorch Dataset over the H5 file produced by build_dataset.py."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize(x: np.ndarray) -> np.ndarray:
    """Per-image zero-mean unit-variance, like the paper's `process_image`."""
    m = x.mean()
    s = x.std()
    return (x - m) / (s + 1e-6)


class SynthFMODataset(Dataset[dict[str, torch.Tensor]]):
    """H5-backed dataset. Returns dict of float32 tensors in (C, H, W) layout."""

    def __init__(self, h5_path: str | Path) -> None:
        self.path = str(h5_path)
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.keys = sorted(k for k in f.keys() if k.startswith("sample_"))
            self.n = len(self.keys)

    def _h(self) -> h5py.File:
        # Open lazily per-worker; h5py is not fork-safe.
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = self._h()[self.keys[idx]]
        img = _normalize(g["image"][()]).astype(np.float32)
        bgr = _normalize(g["bgr"][()]).astype(np.float32)
        tdf = g["tdf"][()].astype(np.float32)
        hm = g["hm"][()].astype(np.float32)
        x = np.concatenate([img, bgr], axis=-1).transpose(2, 0, 1)  # (6, H, W)
        return {
            "x": torch.from_numpy(x),
            "tdf": torch.from_numpy(tdf).unsqueeze(0),
            "hm": torch.from_numpy(hm).unsqueeze(0),
        }
