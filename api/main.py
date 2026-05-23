"""FastAPI backend for FMODetect-v2.

Endpoints:
  GET  /health
  GET  /info                       — model info, ckpt path, device
  POST /infer/image                — multipart {image, background} -> JSON {overlay_url, tdf_url}
  POST /infer/video                — multipart {video} -> JSON {overlay_url, frames}
  GET  /static/{filename}          — serves produced outputs
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.fmodetect.inference.runner import infer_pair, infer_video, load_model
from src.fmodetect.inference.trajectory import trajectory_as_dict

CKPT_ENV = "FMODETECT_CKPT"
STATIC_DIR = Path(os.environ.get("FMODETECT_STATIC", "api/_static")).resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FMODetect-v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_state: dict = {"model": None, "device": None, "ckpt": None}


def _ensure_model() -> tuple[torch.nn.Module, torch.device]:
    if _state["model"] is None:
        ckpt = os.environ.get(CKPT_ENV)
        if not ckpt or not Path(ckpt).exists():
            raise HTTPException(503, f"checkpoint not configured (set {CKPT_ENV})")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _state["model"] = load_model(Path(ckpt), device)
        _state["device"] = device
        _state["ckpt"] = ckpt
    return _state["model"], _state["device"]


@app.get("/health")
def health() -> dict:
    return {"ok": True, "cuda": torch.cuda.is_available()}


@app.get("/info")
def info() -> dict:
    try:
        _ensure_model()
    except HTTPException:
        pass
    return {
        "ckpt": _state.get("ckpt"),
        "device": str(_state.get("device")),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _save_upload(up: UploadFile, suffix: str) -> Path:
    p = STATIC_DIR / f"{uuid.uuid4().hex}{suffix}"
    with p.open("wb") as f:
        f.write(up.file.read())
    return p


@app.post("/infer/image")
def infer_image(image: UploadFile = File(...), background: UploadFile = File(...)) -> JSONResponse:
    model, device = _ensure_model()
    ip = _save_upload(image, Path(image.filename or ".png").suffix or ".png")
    bp = _save_upload(background, Path(background.filename or ".png").suffix or ".png")
    img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    bg = cv2.cvtColor(cv2.imread(str(bp)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    res = infer_pair(model, img, bg, device=device)
    uid = uuid.uuid4().hex
    overlay_path = STATIC_DIR / f"{uid}_overlay.png"
    tdf_path = STATIC_DIR / f"{uid}_tdf.png"
    cv2.imwrite(str(overlay_path), res.overlay)
    cv2.imwrite(str(tdf_path), (np.clip(res.tdf, 0, 1) * 255).astype(np.uint8))
    return JSONResponse({
        "overlay_url": f"/static/{overlay_path.name}",
        "tdf_url": f"/static/{tdf_path.name}",
        "trajectories": [trajectory_as_dict(t) for t in res.trajectories],
        "n_detections": len(res.trajectories),
    })


@app.post("/infer/video")
def infer_video_endpoint(video: UploadFile = File(...)) -> JSONResponse:
    model, device = _ensure_model()
    vp = _save_upload(video, Path(video.filename or ".mp4").suffix or ".mp4")
    uid = uuid.uuid4().hex
    out = STATIC_DIR / f"{uid}_overlay.mp4"
    info = infer_video(model, vp, out, device=device)
    return JSONResponse({"overlay_url": f"/static/{out.name}", **info})
