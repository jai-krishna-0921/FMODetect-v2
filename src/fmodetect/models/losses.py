"""FMODetect-v2 losses.

- `tdf_l1`: paper-faithful baseline. L1 on TDF split into D>0 and D=0 supports.
- `tdf_uncertainty_boundary`: novelty. Gaussian-NLL with learned log-variance,
  plus an L1 penalty on the gradient magnitude of the TDF (boundary loss) for
  sharper trajectory endpoints. Reduces to the paper loss when log_var = 0
  and boundary weight = 0.
- `matting_bce`: BCE on the joint matting head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def tdf_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Paper baseline loss: mean L1 on support + mean L1 on non-support."""
    pos = gt > 0
    neg = ~pos
    loss_pos = (pred[pos] - gt[pos]).abs().mean() if pos.any() else pred.new_zeros(())
    loss_neg = pred[neg].abs().mean() if neg.any() else pred.new_zeros(())
    return loss_pos + loss_neg


def _spatial_grad(x: torch.Tensor) -> torch.Tensor:
    """Sobel-like first-order spatial gradients. Returns (B, 2*C, H, W)."""
    # x: (B, C, H, W). Pad replicate.
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2]
    gy = xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1]
    return torch.cat([gx, gy], dim=1)


def tdf_uncertainty_boundary(
    pred: torch.Tensor,
    gt: torch.Tensor,
    log_var: torch.Tensor,
    *,
    boundary_weight: float = 0.1,
    log_var_reg: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Gaussian-NLL on TDF + L1 boundary regularizer.

    NLL = 0.5 * exp(-log_var) * (pred - gt)^2 + 0.5 * log_var
    Split, like the paper, into D>0 and D=0 supports to keep balance.
    """
    pos = gt > 0
    neg = ~pos

    # NLL (Kendall & Gal 2017). Compute in fp32 for AMP stability;
    # clamp log_var so exp(-log_var) cannot overflow fp16.
    log_var = log_var.float().clamp(-6.0, 6.0)
    pred_f = pred.float()
    gt_f = gt.float()
    inv_var = torch.exp(-log_var)
    sq = (pred_f - gt_f) ** 2
    nll = 0.5 * inv_var * sq + 0.5 * log_var
    nll_pos = nll[pos].mean() if pos.any() else nll.new_zeros(())
    nll_neg = nll[neg].mean() if neg.any() else nll.new_zeros(())
    nll_total = nll_pos + nll_neg

    # Boundary L1 on gradients
    g_pred = _spatial_grad(pred)
    g_gt = _spatial_grad(gt)
    boundary = (g_pred - g_gt).abs().mean()

    # Mild regularizer on log_var to discourage drift
    reg = (log_var**2).mean()

    total = nll_total + boundary_weight * boundary + log_var_reg * reg
    return {
        "tdf_loss": total,
        "tdf_nll": nll_total.detach(),
        "tdf_boundary": boundary.detach(),
        "tdf_logvar_reg": reg.detach(),
    }


def matting_bce(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits for the H*M (blurred mask) head. AMP-safe."""
    return F.binary_cross_entropy_with_logits(logits, gt.clamp(0.0, 1.0))


def combined_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    matting_weight: float = 0.5,
    boundary_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Sum of TDF (uncertainty+boundary) and matting (BCE) losses."""
    parts = tdf_uncertainty_boundary(
        out["tdf"], targets["tdf"], out["log_var_tdf"],
        boundary_weight=boundary_weight,
    )
    total = parts["tdf_loss"]
    if "hm_logits" in out and "hm" in targets:
        m_loss = matting_bce(out["hm_logits"], targets["hm"])
        parts["matting_bce"] = m_loss.detach()
        total = total + matting_weight * m_loss
    parts["total"] = total
    return parts
