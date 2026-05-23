# Vast.ai resume training — workflow

Resumes FMODetect-v2 training from the saved Colab checkpoint (epoch 28, val −3.76) to ~epoch 45 in ~45 min on an RTX PRO 6000 (96 GB).

## One-time prep (do this *before* renting the GPU; meter is not running yet)

### 1. Make the two Drive files shareable
On Google Drive:
- Right-click `My Drive/FMODetect-v2/datasets/synth/vot_fmo.h5` → **Share** → "Anyone with the link" → Done. Copy the URL, extract the **file ID** (the 33-character chunk between `/d/` and `/view`).
- Same for `My Drive/FMODetect-v2/experiments/checkpoints/run_*/best.pt`.

You should now have two IDs, e.g.:
```
H5_DRIVE_ID   = 1AbCdEfGhIjKlMnOpQrStUvWxYz0123456
CKPT_DRIVE_ID = 1ZyXwVuTsRqPoNmLkJiHgFeDcBa9876543
```

### 2. Rent the box

On Vast.ai, search for **RTX PRO 6000** (or any 80 GB+ card). Filter for:
- **CUDA ≥ 12.4**
- **Internet down ≥ 500 Mbps** (16 GB H5 needs to come down fast)
- **Storage ≥ 100 GB** (H5 + checkpoints + headroom)
- Template: any **PyTorch CUDA** template (they all have Jupyter pre-installed)

Click **RENT**. Wait for `RUNNING` status.

### 3. Choose your path

#### A. Jupyter (recommended — visual progress)

1. Open the **Jupyter** tab on your instance's row.
2. Upload `notebooks/train_vast.ipynb` from this repo, OR run inside the cloned repo.
3. Edit cell 0: paste your two Drive IDs.
4. Run cells top-to-bottom.

#### B. SSH (faster, single command)

```bash
# 1. Connect (Vast gives you the ssh command)
ssh -p <port> root@<ip>

# 2. Run the bootstrap (replace the two IDs)
curl -sL https://raw.githubusercontent.com/jai-krishna-0921/FMODetect-v2/main/scripts/setup_vast.sh -o setup_vast.sh
chmod +x setup_vast.sh
H5_DRIVE_ID=<your_h5_id> CKPT_DRIVE_ID=<your_ckpt_id> bash setup_vast.sh

# 3. When it finishes, scp the tarball back
exit
scp -P <port> root@<ip>:/workspace/experiments_*.tar.gz ./
```

## What happens

| Step | Time | Cost @ $1.20/h |
|---|---|---|
| Container boot + pip + apt | ~3 min | $0.06 |
| `gdown` vot_fmo.h5 (16 GB) | ~5–10 min | $0.10–0.20 |
| `gdown` best.pt (70 MB) | ~5 s | — |
| Resume train epoch 29 → 45 (16 epochs × ~80 s) | ~25 min | $0.50 |
| Smoke inference on example | <1 min | — |
| Pack + download tarball | ~2 min | $0.04 |
| **Total** | **~45 min** | **~$0.90** |

## After

Extract the tarball locally:
```bash
tar xzf experiments_*.tar.gz   # → experiments/checkpoints/run_*/best.pt
```

Then locally:
```bash
PYTHONPATH=. .venv/bin/python scripts/infer.py \
  --ckpt experiments/checkpoints/run_*/best.pt \
  --image FMODetect-master/example/ex1_im.png \
  --bgr   FMODetect-master/example/ex1_bgr.png \
  --out final_overlay.png
```

## Knobs you can tweak

In `configs/vast.yaml` (or via the notebook cell):

- `train.epochs`: total target epoch. Resume picks up at `start_epoch+1`, so `epochs=45` from `best.pt@28` = 16 more epochs.
- `train.batch_size`: 64 fits comfortably on 96 GB. 32 if you're on a smaller card.
- `train.early_stop_patience`: 8 is aggressive — if val plateaus for 8 epochs, stop. Drop to 5 for tighter budget.
- `train.lr`: 2e-5 is the paper's lr. If resuming from a checkpoint that's already low, lower to 1e-5 for fine-tuning.
