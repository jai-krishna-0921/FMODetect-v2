# FMODetect-v2

Modernized PyTorch port of **FMODetect** (Rozumnyi et al., 2021, [arxiv 2012.08216](https://arxiv.org/abs/2012.08216)) with three novelty axes:

1. **CBAM attention** at every U-Net block — channel + spatial gating, ~+0.1M params per stage.
2. **Joint TDF + matting multi-task head** — collapses the paper's stage-1 (detection) and stage-2 (matting) into a single shared-encoder network with two decoders. Reduces the 3-stage pipeline to 2 stages (detection-matting + optional ADMM deblurring).
3. **Uncertainty-weighted boundary loss** — Gaussian-NLL on the Truncated Distance Function with per-pixel learned σ, plus an L1 penalty on `‖∇D - ∇D̂‖` for sharper trajectory endpoints. Reduces to the paper's L1 split-loss when `log_var = 0` and `boundary_weight = 0`.

Original TensorFlow 1.x/2.x code is preserved under [`FMODetect-master/`](FMODetect-master/) for reference.

## Hardware target
Single NVIDIA GTX 1650 (4 GB VRAM, Turing CC 7.5). Trains with `batch_size=2 × grad_accum=8`, AMP fp16, channels-last.

## Project layout

```
src/fmodetect/
├── models/
│   ├── attention.py      CBAM (channel + spatial attention)
│   ├── unet.py           FMODetectNet (encoder + TDF decoder + matting decoder)
│   └── losses.py         tdf_l1 (paper), tdf_uncertainty_boundary (ours), combined_loss
├── data/
│   ├── patterns.py       Procedural pillow-based foreground patterns
│   ├── synthesize.py     I = H*F + (1 - H*M)*B image formation model
│   ├── build_dataset.py  Writes H5 of synthetic VOT-FMO samples
│   └── dataset.py        PyTorch Dataset over H5
├── training/loop.py      AMP + grad-accum + MLflow + TB + ClearML training loop
├── inference/runner.py   load_model, infer_pair, infer_video, colorize_tdf
└── utils/config.py       pydantic-validated YAML config

api/main.py               FastAPI: /infer/image, /infer/video, /info, /health
ui/                       Next.js 15 + React 19 + Tailwind 3 frontend
scripts/                  train.py, infer.py, download_datasets.sh
configs/default.yaml      4-GB-friendly defaults (batch 2, grad accum 8)
tests/                    pytest smoke tests
```

## Setup

```bash
# Python 3.14 + uv (verified torch 2.12.0+cu126 has cp314 wheels)
uv sync --python 3.14

# Datasets (run once; TbD is 25 GB and takes hours)
bash scripts/download_datasets.sh all

# Smoke tests
.venv/bin/python -m pytest tests/ -q
```

## Build the synthetic training set

```bash
# Generates 100 pattern PNGs if datasets/patterns/ is empty
.venv/bin/python -m src.fmodetect.data.build_dataset \
  --bg datasets/vot2016 \
  --patterns datasets/patterns \
  --out datasets/synth/vot_fmo.h5 \
  --n 5000
```

## Train

```bash
.venv/bin/python scripts/train.py --config configs/default.yaml
# Logs: experiments/mlruns/  experiments/tb/  experiments/checkpoints/<run>/best.pt
```

## Infer

```bash
# Single image + background
.venv/bin/python scripts/infer.py \
  --ckpt experiments/checkpoints/<run>/best.pt \
  --image FMODetect-master/example/ex1_im.png \
  --bgr   FMODetect-master/example/ex1_bgr.png \
  --out out.png

# Video (rolling 3-frame median background)
.venv/bin/python scripts/infer.py \
  --ckpt experiments/checkpoints/<run>/best.pt \
  --video FMODetect-master/example/falling_pen.avi \
  --out detections.mp4
```

## Run the demo UI

```bash
# Terminal 1: FastAPI backend
FMODETECT_CKPT=experiments/checkpoints/<run>/best.pt \
  .venv/bin/uvicorn api.main:app --reload --port 8000

# Terminal 2: Next.js frontend
cd ui && npm install && npm run dev    # http://localhost:3000
```

## Status

Code is scaffolded and the data pipeline + model + losses + training loop + inference + API + UI are all in place. Not yet trained.

| Stage | Status |
|------|------|
| Paper read + original code analysis | ✅ |
| PyTorch port (model, losses, dataset) | ✅ |
| Training loop (AMP, grad accum, MLflow, ClearML, TB) | ✅ |
| Inference CLI + FastAPI + Next.js UI | ✅ |
| Smoke tests | ✅ (data + model) |
| Datasets downloaded | ⏳ (background) |
| Synthetic VOT-FMO H5 built | ⏳ |
| Model trained | ⏳ |
| Benchmark numbers | ⏳ |
