---
title: FMODetect v2
emoji: 🌀
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Fast-moving-object detection from a single blurred frame
---

# FMODetect v2 — research demo

PyTorch re-implementation of [FMODetect (Rozumnyi et al., ICCV 2021)](https://arxiv.org/abs/2012.08216)
with three additions: CBAM attention, a joint TDF + matting head, and an
uncertainty-weighted boundary loss.

Source: <https://github.com/jai-krishna-0921/FMODetect-v2>

## How this Space is built

This Space contains only a `Dockerfile`, a `requirements.txt` and this README.
At build time the Dockerfile clones the source repo, builds the Next.js UI as
a static export, and serves it from FastAPI on port 7860.

## Environment

Set these in the Space settings → Variables and secrets:

| key                      | value                                  |
|--------------------------|----------------------------------------|
| `FMODETECT_HF_REPO`      | `<your-username>/fmodetect-v2`         |
| `FMODETECT_HF_FILENAME`  | `best.pt` (default; only set to override) |

The checkpoint is downloaded once at first request and cached on the Space
disk. To swap models, upload a new file to the HF Hub model repo and restart
the Space.

## Hardware

CPU Basic (free) runs inference in ~2–3 s per image pair. T4 small (~$0.40/hr)
brings it under 200 ms.
