# Deploying to Hugging Face Spaces

End result: a public URL like `https://huggingface.co/spaces/<you>/fmodetect-v2`
running the Next.js UI + FastAPI inference on a free CPU tier.

The Space repo is intentionally tiny — just `Dockerfile`, `requirements.txt`,
`README.md`. The Dockerfile clones this GitHub repo at build time, builds the
UI, and runs uvicorn on port 7860. The model checkpoint is pulled from a
Hugging Face model repo on first request.

---

## One-time setup

### 1. Push the checkpoint to a model repo

```bash
# from a machine that has best.pt
huggingface-cli login                              # paste a write token
huggingface-cli repo create fmodetect-v2 --type model
huggingface-cli upload <you>/fmodetect-v2 \
    experiments/checkpoints/run_20260523_085253/best.pt best.pt
```

`<you>` is your HF username. The file ends up at
`https://huggingface.co/<you>/fmodetect-v2/blob/main/best.pt`.

### 2. Create the Space

```bash
huggingface-cli repo create fmodetect-v2 --type space --space_sdk docker
git clone https://huggingface.co/spaces/<you>/fmodetect-v2 space-repo
cp space/Dockerfile space/requirements.txt space/README.md space-repo/
cd space-repo
git add . && git commit -m "Initial Space" && git push
```

The Space starts building; first build takes ~6–8 min (torch wheel
download dominates).

### 3. Wire the checkpoint env vars

In the Space UI → Settings → "Variables and secrets":

| key                | value                  |
|--------------------|------------------------|
| `FMODETECT_HF_REPO`| `<you>/fmodetect-v2`   |

Restart the Space (Settings → Factory reboot is unnecessary; "Restart Space"
is enough).

---

## Updating

- **UI / API code change** → push to GitHub `main`, then Restart the Space.
  The Dockerfile clones fresh on each build.
- **New checkpoint** → upload to the HF model repo (same `best.pt` path),
  then Restart the Space. Cache lives at `/data/hf-cache` so it'll fetch the
  new revision.
- **Dependencies change** → also edit `space/requirements.txt` and push to
  the Space repo.

---

## Local equivalent (sanity check before pushing)

```bash
# build + run the same image locally
docker build --build-arg REPO_REF=main -t fmodetect:local space/
docker run --rm -p 7860:7860 \
  -e FMODETECT_HF_REPO=<you>/fmodetect-v2 \
  fmodetect:local
# then open http://localhost:7860
```

---

## Costs

| tier                  | latency / pair | $ / hr | when |
|-----------------------|----------------|--------|------|
| CPU basic (free)      | 2–3 s          | 0.00   | default |
| Nvidia T4 small (paid)| ~150 ms        | 0.40   | if traffic > demo level |
