"""FastAPI backend for FMODetect-v2.

Endpoints:
  GET  /health
  GET  /info                       — model info, ckpt path, device
  GET  /examples                   — list bundled demo image pairs
  POST /infer/sample/{name}        — run inference on a bundled example
  POST /infer/image                — multipart {image, background} -> JSON
  POST /infer/video                — multipart {video} -> JSON
  GET  /static/{filename}          — serves produced outputs
  GET  /examples/{name}/{file}     — serves bundled example assets
"""
from __future__ import annotations

import json
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
EXAMPLES_DIR = Path(__file__).resolve().parent / "_examples"

app = FastAPI(title="FMODetect-v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if EXAMPLES_DIR.exists():
    app.mount("/examples-static", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples-static")

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
    ip = _save_upload(image, Path(image.filename or ".png").suffix or ".png")
    bp = _save_upload(background, Path(background.filename or ".png").suffix or ".png")
    return JSONResponse(_run_pair(ip, bp))


def _run_pair(img_path: Path, bg_path: Path) -> dict:
    model, device = _ensure_model()
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    res = infer_pair(model, img, bg, device=device)
    uid = uuid.uuid4().hex
    overlay_path = STATIC_DIR / f"{uid}_overlay.png"
    tdf_path = STATIC_DIR / f"{uid}_tdf.png"
    cv2.imwrite(str(overlay_path), res.overlay)
    cv2.imwrite(str(tdf_path), (np.clip(res.tdf, 0, 1) * 255).astype(np.uint8))
    return {
        "overlay_url": f"/static/{overlay_path.name}",
        "tdf_url": f"/static/{tdf_path.name}",
        "trajectories": [trajectory_as_dict(t) for t in res.trajectories],
        "n_detections": len(res.trajectories),
    }


@app.get("/examples")
def list_examples() -> list[dict]:
    manifest = EXAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return []
    items = json.loads(manifest.read_text())
    return [
        {**it, "thumb_url": f"/examples-static/{it['name']}/thumb.jpg"}
        for it in items
        if (EXAMPLES_DIR / it["name"] / "thumb.jpg").exists()
    ]


@app.post("/infer/sample/{name}")
def infer_sample(name: str) -> JSONResponse:
    d = EXAMPLES_DIR / name
    img, bg = d / "fmo.png", d / "bg.png"
    if not (img.exists() and bg.exists()):
        raise HTTPException(404, f"example '{name}' not found")
    return JSONResponse(_run_pair(img, bg))


@app.post("/infer/video")
def infer_video_endpoint(video: UploadFile = File(...)) -> JSONResponse:
    model, device = _ensure_model()
    vp = _save_upload(video, Path(video.filename or ".mp4").suffix or ".mp4")
    uid = uuid.uuid4().hex
    out = STATIC_DIR / f"{uid}_overlay.mp4"
    info = infer_video(model, vp, out, device=device)
    return JSONResponse({"overlay_url": f"/static/{out.name}", **info})
