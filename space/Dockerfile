# Hugging Face Space Dockerfile for FMODetect v2.
# Builds the Next.js UI as a static export, then runs FastAPI on port 7860.
# The source repo is cloned at build time so the Space stays tiny.

ARG REPO_URL=https://github.com/jai-krishna-0921/FMODetect-v2.git
ARG REPO_REF=main

# --- Stage 1: build the Next.js static export ---
FROM node:20-bookworm-slim AS ui-build
ARG REPO_URL
ARG REPO_REF
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} app
WORKDIR /src/app/ui
RUN npm install --no-audit --no-fund
RUN NEXT_OUTPUT=export npm run build

# --- Stage 2: Python runtime serving API + static UI ---
FROM python:3.12-slim AS runtime
ARG REPO_URL
ARG REPO_REF

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    FMODETECT_STATIC=/data/static \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone source at build time (overlaid in the next step with the UI build).
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app && \
    rm -rf /app/ui && mkdir -p /app/ui

# Bring in the built static export from stage 1.
COPY --from=ui-build /src/app/ui/out /app/ui/out

# Install deploy-only requirements (CPU torch).
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Writable runtime dirs for HF cache and generated overlays.
RUN mkdir -p /data/hf-cache /data/static && chmod -R 777 /data

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
