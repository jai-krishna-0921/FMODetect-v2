# Experiments log

Honest record of training runs so future iterations don't repeat mistakes.

## v2 — first viable run (2026-05-23)

- **Config:** `configs/colab.yaml`, synth v1 (radius 20–60 px, square patterns), 120 epochs.
- **Box:** Colab T4 + later resumed on Vast (RTX PRO 6000), bs 16 then 64.
- **Result:**
  - val_total: **−4.89**
  - falling P/R: **0.067 / 0.091** (paper baseline: 0.989 / 0.825 — ~9× gap)
  - Real detections present but rare; TDF peaks correctly on motion-blurred regions.
- **Ckpt:** `experiments/checkpoints/run_20260523_085253/best.pt` (70 MB). Mirrored to HF Hub `jai-krishna/fmodetect-v2`. **Deployed.**

## v3 — synth-distribution change, regressed (2026-05-24)

- **Hypothesis:** the 9× recall gap is a synth↔real distribution mismatch; v1's 20–60 px discs don't match falling-dataset objects (5–15 px, elongated). Broaden radius to (8, 50) and add aspect-ratio jitter (0.3, 1.0).
- **Config:** `configs/vast_rerun.yaml`, synth v3, planned 500 epochs.
- **Box:** Vast RTX PRO 6000 Blackwell (1× $1.137/h, CUDA 12.8, 96 GB VRAM).
- **What happened:**
  - val_total peaked at epoch 43 (**−3.31**), then plateaued; early-stopping triggered at epoch 83.
  - falling P/R: **0.000 / 0.000** — strictly worse than v2.
  - Model produces 1–3 trajectories per frame but in the **wrong location** (no overlap with GT bbox even at threshold 0.05).
- **Ckpt:** `experiments/checkpoints/v3_vast/best.pt` (archived — not deployed).
- **Cost:** ~$5 of $11.65 credit. Sunk.

### Why v3 regressed (best guess)

Two synth changes at once (broader radii AND aspect jitter) plus only 83 effective epochs. Can't tell which change broke things. The model learned to fire TDF peaks but not to localize them on real frames — classic synth-real generalization failure when the synth distribution shifts faster than the model can re-learn localization.

### Setup hiccups worth remembering for next Vast run

- **`download.pytorch.org` 403** intermittently from this Vast pool — direct R2 wheel URL works (`https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl`).
- **`releases.astral.sh` (uv installer) SSL hangs** from Vast — fall back to plain `pip install` of a pinned wheel.
- **VOT2016 download is throttled** by votchallenge.net; 8-way parallel only gives ~2× speedup. Worth caching VOT2016 on HF Datasets for future runs.
- **dpkg interrupted state** at container start — run `dpkg --configure -a` before any `apt-get install`.
- **Training loop has no checkpoint pruning** — at `save_every_n_epochs=3` over 500 ep that's 12 GB of ckpts. Bumped to 6 for the rerun.

## Lessons → next experiment design rules

1. **Change one variable at a time.** Don't combine radius range + aspect ratio + epochs + LR. If v3 had only changed radius, we'd know whether the aspect jitter was the culprit.
2. **Floor: always re-eval on real falling before claiming progress.** synth val is not a proxy for real performance once the distribution shifts.
3. **Cheap experiment first.** Before another synth change, run v2 distribution for 300–500 ep (~$3). If recall improves with pure capacity, the bottleneck is undertraining, not distribution.
4. **Mirror VOT2016 to HF Datasets** as a one-time chore — saves ~15 min of slow Czech-server pulls on every new box.
5. **Bake a checkpoint-pruning step into `training/loop.py`** before any long run — disk is the silent killer on rented boxes.
