#!/usr/bin/env bash
# Vast.ai bootstrap — run this AS SOON AS the container shells you in.
# Assumes a base PyTorch image (most Vast templates already have torch+cuda).
#
# Steps:
#   1. apt-get + pip system deps
#   2. git clone the repo
#   3. gdown the H5 + best.pt from your Drive (you provide the file IDs)
#   4. resume training (15 more epochs from epoch 28)
#   5. run eval on whatever local eval data exists
#   6. tar up the experiments/ dir for download via Vast's web UI
#
# Usage on the Vast box (after ssh):
#   curl -sL https://raw.githubusercontent.com/jai-krishna-0921/FMODetect-v2/main/scripts/setup_vast.sh -o setup_vast.sh
#   chmod +x setup_vast.sh
#   H5_DRIVE_ID=<...> CKPT_DRIVE_ID=<...> bash setup_vast.sh
#
# Time budget on an RTX PRO 6000 ($1.20/h):
#   apt/pip + clone:            ~3 min
#   gdown H5 (~16 GB):          ~5-10 min
#   gdown best.pt (~70 MB):     ~5 s
#   resume train 15 epochs:    ~25 min @ bs 64
#   eval + pack:                ~5 min
#   TOTAL:                     ~45-55 min  ≈  $0.90-1.10
set -euo pipefail

: "${H5_DRIVE_ID:?Set H5_DRIVE_ID to the Drive file ID for vot_fmo.h5}"
: "${CKPT_DRIVE_ID:?Set CKPT_DRIVE_ID to the Drive file ID for best.pt}"
: "${RESUME_TOTAL_EPOCHS:=45}"

WORKDIR=/workspace
REPO_URL=https://github.com/jai-krishna-0921/FMODetect-v2.git
REPO_DIR=${WORKDIR}/FMODetect-v2

mkdir -p ${WORKDIR}/{data,experiments/checkpoints/resume,experiments/mlruns,experiments/tb}

# --- 1. System deps ---
echo "==> apt + pip deps"
apt-get update -qq && apt-get install -y -qq git curl unzip ffmpeg >/dev/null
pip install -q --upgrade pip
pip install -q gdown scikit-image h5py opencv-python-headless scipy tqdm \
               imageio[ffmpeg] mlflow clearml pydantic pyyaml \
               typer rich tensorboard

# --- 2. Repo ---
echo "==> clone repo"
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi

# --- 3. Pull dataset + checkpoint from Drive ---
echo "==> gdown vot_fmo.h5 (~16 GB)"
gdown --id "$H5_DRIVE_ID" -O ${WORKDIR}/data/vot_fmo.h5
ls -lh ${WORKDIR}/data/vot_fmo.h5

echo "==> gdown best.pt"
gdown --id "$CKPT_DRIVE_ID" -O ${WORKDIR}/experiments/checkpoints/resume/best.pt
ls -lh ${WORKDIR}/experiments/checkpoints/resume/best.pt

# --- 4. Train ---
echo "==> resume training"
cd "$REPO_DIR"
PYTHONPATH=. PYTHONUNBUFFERED=1 python -u scripts/train.py \
    --config configs/vast.yaml \
    --resume ${WORKDIR}/experiments/checkpoints/resume/best.pt \
    --epochs ${RESUME_TOTAL_EPOCHS}

# --- 5. Pack outputs ---
echo "==> pack outputs"
LATEST=$(ls -td ${WORKDIR}/experiments/checkpoints/run_* | head -1)
echo "Latest run dir: $LATEST"
cd ${WORKDIR}
tar czf experiments_$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='*.npy' \
    experiments/

echo ""
echo "DONE."
echo "Download the tarball via Vast's web UI from:"
echo "  ${WORKDIR}/experiments_*.tar.gz"
echo "It contains: trained checkpoint(s) + mlflow run + tensorboard logs."
