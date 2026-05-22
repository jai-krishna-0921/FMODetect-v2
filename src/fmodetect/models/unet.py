"""FMODetect-v2 U-Net with CBAM attention and joint TDF + matting heads.

Architecture mirrors the original FMODetect detection net (3-stage encoder blocks
with LeakyReLU(0.1), 4 max-pool downsamples, transposed-conv decoder with skips)
but adds:
  - CBAM after every conv block (novelty #1)
  - A second decoder head that predicts blurred mask H*M (novelty #2 — multi-task)
  - A learned log-variance map alongside the TDF (novelty #3 — uncertainty)

Inputs: image I and median background B concatenated along channels (6 ch).
Outputs: dict with keys
    'tdf'         : [B, 1, H, W]   truncated distance function in [0, 1] (sigmoid)
    'log_var_tdf' : [B, 1, H, W]   per-pixel log-variance for the TDF head
    'hm_logits'   : [B, 1, H, W]   matting head raw logits (use with BCE-with-logits)
    'hm'          : [B, 1, H, W]   sigmoid(hm_logits) — for inference/display only
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import CBAM


def _conv_block(in_ch: int, out_ch: int, *, use_cbam: bool) -> nn.Sequential:
    # Three conv-LeakyReLU layers, matching the original `conv_layer` from net_model.py.
    layers: list[nn.Module] = []
    for i in range(3):
        c_in = in_ch if i == 0 else out_ch
        layers += [
            nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        ]
    if use_cbam:
        layers.append(CBAM(out_ch))
    return nn.Sequential(*layers)


@dataclass
class UNetConfig:
    in_channels: int = 6  # image (3) + median bgr (3)
    base_channels: tuple[int, int, int, int, int] = (16, 64, 128, 256, 256)
    use_cbam: bool = True
    predict_matting: bool = True
    predict_uncertainty: bool = True


class FMODetectNet(nn.Module):
    def __init__(self, cfg: UNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UNetConfig()
        c1, c2, c3, c4, cb = self.cfg.base_channels

        # ---- encoder ----
        self.enc1 = _conv_block(self.cfg.in_channels, c1, use_cbam=self.cfg.use_cbam)
        self.enc2 = _conv_block(c1, c2, use_cbam=self.cfg.use_cbam)
        self.enc3 = _conv_block(c2, c3, use_cbam=self.cfg.use_cbam)
        self.enc4 = _conv_block(c3, c4, use_cbam=self.cfg.use_cbam)
        self.bottleneck = _conv_block(c4, cb, use_cbam=self.cfg.use_cbam)
        self.pool = nn.MaxPool2d(2)

        # ---- TDF decoder (always present) ----
        self.tdf_dec = _Decoder(cb, c4, c3, c2, c1, use_cbam=self.cfg.use_cbam)
        # Head: TDF + (optional) log-variance, stacked over the 4->4->1 trailing convs from the paper.
        out_channels = 2 if self.cfg.predict_uncertainty else 1
        self.tdf_head = nn.Sequential(
            nn.Conv2d(c1, 4, 3, padding=1),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.Conv2d(4, out_channels, 3, padding=1),
        )

        # ---- matting decoder (novelty: shares encoder) ----
        if self.cfg.predict_matting:
            self.mat_dec = _Decoder(cb, c4, c3, c2, c1, use_cbam=self.cfg.use_cbam)
            self.mat_head = nn.Sequential(
                nn.Conv2d(c1, 4, 3, padding=1),
                nn.Conv2d(4, 1, 3, padding=1),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))

        skips = (s4, s3, s2, s1)

        tdf_feat = self.tdf_dec(b, skips)
        tdf_out = self.tdf_head(tdf_feat)
        out: dict[str, torch.Tensor] = {}
        if self.cfg.predict_uncertainty:
            out["tdf"] = torch.sigmoid(tdf_out[:, :1])
            out["log_var_tdf"] = tdf_out[:, 1:2]
        else:
            out["tdf"] = torch.sigmoid(tdf_out)

        if self.cfg.predict_matting:
            mat_feat = self.mat_dec(b, skips)
            hm_logits = self.mat_head(mat_feat)
            out["hm_logits"] = hm_logits
            out["hm"] = torch.sigmoid(hm_logits)

        return out


class _Decoder(nn.Module):
    def __init__(self, cb: int, c4: int, c3: int, c2: int, c1: int, *, use_cbam: bool) -> None:
        super().__init__()
        self.up4 = nn.ConvTranspose2d(cb, c3, 2, stride=2)
        self.dec4 = _conv_block(c3 + c4, c3, use_cbam=use_cbam)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3 = _conv_block(c2 + c3, c2, use_cbam=use_cbam)
        self.up2 = nn.ConvTranspose2d(c2, max(c2 // 2, 32), 2, stride=2)
        self.dec2 = _conv_block(max(c2 // 2, 32) + c2, max(c2 // 2, 32), use_cbam=use_cbam)
        self.up1 = nn.ConvTranspose2d(max(c2 // 2, 32), c1, 2, stride=2)
        self.dec1 = _conv_block(c1 + c1, c1, use_cbam=use_cbam)

    def forward(self, b: torch.Tensor, skips: tuple[torch.Tensor, ...]) -> torch.Tensor:
        s4, s3, s2, s1 = skips
        x = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return x
