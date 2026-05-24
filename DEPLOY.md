# Deploying to Hugging Face Spaces

End result: a public URL like `https://huggingface.co/spaces/<you>/fmodetect-v2`
running the Next.js UI + FastAPI inference on a free CPU tier.

The Space repo is intentionally tiny — just `Dockerfile`, `requirements.txt`,
`README.md`. The Dockerfile clones this GitHub repo at build time, builds the
UI, and runs uvicorn on port 7860. The model checkpoint is pulled from a
Hugging Face model repo on first request.

The live deployment of this repo is at
**<https://huggingface.co/spaces/jai-krishna/fmodetect-v2>** — substitute your
own username throughout when bootstrapping a fork.

---

## One-time setup

### 0. Install the CLI (once)

```bash
VIRTUAL_ENV=.venv uv pip install -U huggingface_hub
.venv/bin/hf --version          # 1.x
```

(The old `huggingface-cli` is deprecated as of 1.x — use `hf`.)

### 1. Push the checkpoint to a model repo

```bash
.venv/bin/hf auth login                            # write token from hf.co/settings/tokens
.venv/bin/hf repos create fmodetect-v2 --type model --public
.venv/bin/hf upload <you>/fmodetect-v2 \
    experiments/checkpoints/run_20260523_085253/best.pt best.pt
```

`<you>` is your HF username. The file ends up at
`https://huggingface.co/<you>/fmodetect-v2/blob/main/best.pt`.

### 2. Create the Space (env preset in one shot)

```bash
.venv/bin/hf repos create fmodetect-v2 \
    --type space --space-sdk docker --public \
    --env FMODETECT_HF_REPO=<you>/fmodetect-v2

git clone https://huggingface.co/spaces/<you>/fmodetect-v2 /tmp/space-repo
cp space/Dockerfile space/requirements.txt space/README.md /tmp/space-repo/
cd /tmp/space-repo
git add . && git commit -m "Initial Space" && git push
```

The Space starts building immediately; first build takes ~6–8 min (CPU torch
wheel dominates). Watch the build log on the Space page.

Because `--env FMODETECT_HF_REPO=…` was passed at creation, the Space already
knows where to pull the checkpoint from — no manual settings step needed.

---

## Updating

- **UI / API code change** → push to GitHub `main`, then Restart the Space
  (Settings → "Restart Space"). The Dockerfile clones fresh on each build.
- **New checkpoint** → upload to the HF model repo (same `best.pt` path),
  then Restart the Space. Cache lives at `/data/hf-cache` so it'll fetch the
  new revision.
- **Dependencies change** → edit `space/requirements.txt`, push to the Space
  repo (not GitHub — the Space repo is separate).

---

## Local equivalent (sanity check before pushing)

```bash
docker build --build-arg REPO_REF=main -t fmodetect:local space/
docker run --rm -p 7860:7860 \
    -e FMODETECT_HF_REPO=<you>/fmodetect-v2 \
    fmodetect:local
# open http://localhost:7860
```

---

## Costs

| tier                  | latency / pair | $ / hr | when |
|-----------------------|----------------|--------|------|
| CPU basic (free)      | 2–3 s          | 0.00   | default |
| Nvidia T4 small (paid)| ~150 ms        | 0.40   | if traffic > demo level |
