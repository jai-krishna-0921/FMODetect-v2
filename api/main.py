"""FastAPI backend for FMODetect-v2.

All application routes live under /api/* so the root path can serve
the built Next.js static export when deployed as a single container.

  GET  /api/health
  GET  /api/info
  GET  /api/examples
  POST /api/infer/sample/{name}
  POST /api/infer/image
  POST /api/infer/video
  GET  /api/static/{filename}            — produced overlays/TDFs
  GET  /api/examples-static/{name}/...   — bundled demo assets
  GET  /                                  — Next.js static export (if present)
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.fmodetect.inference.runner import infer_pair, infer_video, load_model
from src.fmodetect.inference.trajectory import trajectory_as_dict

CKPT_ENV = "FMODETECT_CKPT"
HF_REPO_ENV = "FMODETECT_HF_REPO"          # e.g. "jai-krishna/fmodetect-v2"
HF_FILENAME_ENV = "FMODETECT_HF_FILENAME"  # defaults to "best.pt"

STATIC_DIR = Path(os.environ.get("FMODETECT_STATIC", "api/_static")).resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR = Path(__file__).resolve().parent / "_examples"
UI_BUILD_DIR = Path(os.environ.get("FMODETECT_UI_DIR", "ui/out")).resolve()

app = FastAPI(title="FMODetect-v2", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if EXAMPLES_DIR.exists():
    app.mount("/api/examples-static", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples-static")

router = APIRouter(prefix="/api")

_state: dict = {"model": None, "device": None, "ckpt": None}


def _resolve_ckpt() -> Path | None:
    """Local FMODETECT_CKPT wins; else try HF Hub if FMODETECT_HF_REPO is set."""
    env_path = os.environ.get(CKPT_ENV)
    if env_path and Path(env_path).exists():
        return Path(env_path)
    repo = os.environ.get(HF_REPO_ENV)
    if repo:
        from huggingface_hub import hf_hub_download
        filename = os.environ.get(HF_FILENAME_ENV, "best.pt")
        return Path(hf_hub_download(repo_id=repo, filename=filename))
    return None


def _ensure_model() -> tuple[torch.nn.Module, torch.device]:
    if _state["model"] is None:
        ckpt = _resolve_ckpt()
        if ckpt is None:
            raise HTTPException(503, f"checkpoint not configured (set {CKPT_ENV} or {HF_REPO_ENV})")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _state["model"] = load_model(ckpt, device)
        _state["device"] = device
        _state["ckpt"] = str(ckpt)
    return _state["model"], _state["device"]


@router.get("/health")
def health() -> dict:
    return {"ok": True, "cuda": torch.cuda.is_available()}


@router.get("/info")
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
        "overlay_url": f"/api/static/{overlay_path.name}",
        "tdf_url": f"/api/static/{tdf_path.name}",
        "trajectories": [trajectory_as_dict(t) for t in res.trajectories],
        "n_detections": len(res.trajectories),
    }


@router.post("/infer/image")
def infer_image(image: UploadFile = File(...), background: UploadFile = File(...)) -> JSONResponse:
    ip = _save_upload(image, Path(image.filename or ".png").suffix or ".png")
    bp = _save_upload(background, Path(background.filename or ".png").suffix or ".png")
    return JSONResponse(_run_pair(ip, bp))


@router.get("/examples")
def list_examples() -> list[dict]:
    manifest = EXAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return []
    items = json.loads(manifest.read_text())
    return [
        {**it, "thumb_url": f"/api/examples-static/{it['name']}/thumb.jpg"}
        for it in items
        if (EXAMPLES_DIR / it["name"] / "thumb.jpg").exists()
    ]


@router.post("/infer/sample/{name}")
def infer_sample(name: str) -> JSONResponse:
    d = EXAMPLES_DIR / name
    img, bg = d / "fmo.png", d / "bg.png"
    if not (img.exists() and bg.exists()):
        raise HTTPException(404, f"example '{name}' not found")
    return JSONResponse(_run_pair(img, bg))


@router.post("/infer/video")
def infer_video_endpoint(video: UploadFile = File(...)) -> JSONResponse:
    model, device = _ensure_model()
    vp = _save_upload(video, Path(video.filename or ".mp4").suffix or ".mp4")
    uid = uuid.uuid4().hex
    out = STATIC_DIR / f"{uid}_overlay.mp4"
    meta = infer_video(model, vp, out, device=device)
    return JSONResponse({"overlay_url": f"/api/static/{out.name}", **meta})


app.include_router(router)

# Serve the Next.js static export at / (production single-container layout).
# In local dev, run `next dev` on port 3000 and leave UI_BUILD_DIR absent.
if UI_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_BUILD_DIR), html=True), name="ui")
