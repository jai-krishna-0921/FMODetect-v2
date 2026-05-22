# FMODetect-v2 UI

Next.js 15 + React 19 + Tailwind 3 frontend for the FastAPI backend.

## Run

```bash
# 1. Backend (FastAPI)
FMODETECT_CKPT=experiments/checkpoints/<run>/best.pt \
  .venv/bin/uvicorn api.main:app --reload --port 8000

# 2. Frontend (Next.js)
cd ui && npm install
npm run dev     # http://localhost:3000
```

The frontend rewrites `/api/*` and `/static/*` to `localhost:8000` (see `next.config.mjs`).
