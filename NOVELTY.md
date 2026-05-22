# Novelty — what's different from the original FMODetect

**Original paper**: Rozumnyi et al., 2021, *FMODetect: Robust Detection of Fast Moving Objects* — arxiv 2012.08216.
TL;DR: a 3-stage pipeline (vanilla U-Net detection → 1-encoder-3-decoder matting/fitting net → ADMM deblurring) that detects FMOs in single frames. Reported 0.97 recall and 0.715 TIoU on TbD with deblurring; 20 fps real-time without it. **Only the detection net is released** in the original repo.

## What FMODetect-v2 changes

### Novelty axis 1 — **CBAM attention** at every U-Net block

| Original | v2 |
|------|------|
| Plain 3-conv block + LeakyReLU(0.1), no normalization, no attention. | Same 3-conv block, **plus a CBAM (Woo et al., ECCV 2018) module** at the end of every encoder + decoder block. |

**Why it helps for FMO detection specifically**
- FMOs are small, low-contrast, blurry streaks in a large frame — the network has to suppress a *huge* number of irrelevant background pixels.
- CBAM's spatial attention gives the network a learned soft-mask over "where to look" *before* the next conv stage. Its channel attention reweights filters depending on whether they're firing on background texture vs. blur-streak texture.
- Adds ~0.1 M params per stage (negligible vs the 4.8 M baseline).

**How we will measure it**: ablation run with `model.use_cbam: false` in `configs/`. Compare precision/recall + TIoU on the TbD eval set.

---

### Novelty axis 2 — **Joint TDF + matting multi-task head** (shared encoder)

| Original | v2 |
|------|------|
| Stage-1 net predicts TDF only (4.8 M params). Stage-2 net is a *separate* 5.7 M-param 1-encoder-3-decoder net that re-encodes per-crop images. **Two full forward passes**, two encoders trained independently. | A single 6.0 M-param net with one shared encoder and **two decoders**: one predicts TDF, one predicts the blurred-mask `H*M`. One forward pass produces both. |

**Why it helps**
- The two tasks are deeply correlated — wherever the TDF says "trajectory is here", the blurred mask `H*M` should also be non-zero (because `H*M = H ⊛ M` and `H` is the trajectory). Multi-task learning with related auxiliary targets is a standard regularizer (e.g. UberNet, MultiNet).
- Eliminates the per-detection-crop second forward pass — the paper's stage-2 ran on `256×256` crops *after* connected-component analysis, which is the main reason their full pipeline drops from 20 fps to 0.4 fps when deblurring is enabled. We get matting "for free".
- Engineering simplification: one checkpoint, one config, one training run.

**How we will measure it**: ablation run with `model.predict_matting: false` removes the matting decoder + loss term. Compare TDF-only TIoU + matting-head IoU on `H*M`.

---

### Novelty axis 3 — **Uncertainty-weighted boundary loss**

| Original | v2 |
|------|------|
| Plain L1 split into D>0 and D=0 supports. Same weight on every pixel. | Replace L1 with **Gaussian-NLL** (Kendall & Gal 2017) with **per-pixel learned log-variance σ²**, plus an **L1 penalty on gradient magnitude** `‖∇D − ∇D̂‖`. Reduces to the paper's loss when σ² = 1, boundary weight = 0. |

**Why it helps**
- *Uncertainty*: the network can downweight ambiguous pixels (trajectory endpoints, regions where the bg estimate is poor) without us having to design heuristic per-pixel weights. The paper's analysis shows their main failure mode is dropped recall at trajectory endpoints — exactly the regions where uncertainty should be highest.
- *Boundary loss*: L1 on the TDF is rotationally insensitive — it can be minimized by a "blurry" estimate. Penalizing `‖∇D − ∇D̂‖` enforces sharp trajectory edges directly. This is a standard trick in image-to-image translation (e.g. pix2pix's gradient-difference loss).

**How we will measure it**: ablation with `boundary_weight: 0.0` and `predict_uncertainty: false` recovers the paper's loss. Compare TDF L1 + trajectory endpoint recall (boundary loss should improve the latter).

---

## What can the model predict?

| Output | Where it comes from |
|---|---|
| **TDF heatmap** — "where is the trajectory" | Direct model output (sigmoid-squashed) |
| **Blurred mask `H*M`** | Direct model output (matting decoder, sigmoid) |
| **Per-pixel uncertainty σ²** | Direct model output (log-variance map) |
| **Parametric trajectory** `C(t) = c0 + c1·t + c2·t²` | Post-hoc fit of TDF skeleton via least squares — `src/fmodetect/inference/trajectory.py` |
| **Speed in px / frame** | Arc length of fitted `C(t)` on `t∈[0,1]` (one exposure ≡ one frame) |
| **Speed in px / sec** | `speed_px_per_frame × fps` (when caller passes `fps`) |
| **Object radius in px** | Median TDF tube half-width along the skeleton |
| **Bounding box** | `regionprops` bbox of the thresholded TDF component |
| **Detection confidence** | Mean TDF value along the skeleton |

**What we do *not* yet predict** (and what's needed to add it):
- *Sharp object appearance `F`*: requires ADMM deblurring (paper's stage 3) — not implemented yet
- *Sharp mask `M`*: same as above
- *3D rotation*: paper extension TbD-3D — not implemented
- *Real-world speed in m/s*: needs camera calibration (focal length, distance to object)

---

## How will we know the changes actually help?

The evaluation harness (`scripts/eval.py`) runs the full inference pipeline (detection → trajectory fit) on each real-world dataset (TbD, TbD-3D, falling) and computes:

| Metric | Definition | Paper baseline |
|---|---|---|
| **Precision** | TP / (TP + FP) where TP = bbox-IoU ≥ 0.1 (paper §4) | 0.879 (TbD-3D, FMODetect) |
| **Recall** | TP / (TP + FN) | 0.835 (TbD-3D), 0.970 (TbD) |
| **TIoU** | Eq. 7 — average IoU of swept disc masks along GT and predicted trajectories | 0.519 (TbD detection-only) |

**Generalization to unseen data**: training is purely synthetic (VOT2016 backgrounds + procedurally generated discs). The paper demonstrates this generalizes well to all three real eval sets *and* YouTube videos with arbitrary object shapes (hand, cap, keychain, tennis ball). We use the same synthetic training scheme — generalization should match or exceed the baseline as long as the loss + architecture changes don't hurt.

**Headline ablation we plan to run** (one big training comparison):

| Variant | CBAM | Multi-task | Uncertainty+Boundary | Expected gain |
|---|---|---|---|---|
| baseline (paper-faithful) | ✗ | ✗ | ✗ | — |
| +CBAM | ✓ | ✗ | ✗ | small recall ↑ |
| +CBAM +Matting | ✓ | ✓ | ✗ | matting head usable |
| **full v2** | ✓ | ✓ | ✓ | TIoU + endpoint recall ↑ |

We *do not yet* claim accuracy beats the paper — claiming that requires the trained model + completed eval runs. This document describes the plan and the contributions.

## Honest assessment

These are **modest, well-motivated incremental changes**, not a paradigm shift. CBAM, uncertainty-weighted regression, boundary-gradient loss, and joint multi-task heads are all established techniques. The novelty is in the **combination and the FMO-specific motivation**, plus the **engineering simplification** (one network instead of two).

A more ambitious change would be a temporal-aware 3-frame input (which we deferred — the user voted against it), or a transformer bottleneck (deferred for VRAM reasons). Those would be plausible follow-ups.
