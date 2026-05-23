#!/usr/bin/env bash
# scripts/vast.sh — drive a Vast.ai box from your local machine.
#
# One-time setup:
#   1. Rent an instance on Vast.ai (add your SSH key first).
#   2. Copy the SSH command Vast shows you (e.g. ssh -p 12345 root@1.2.3.4) into ~/.vastrc, OR
#      export VAST_HOST and VAST_PORT in your shell:
#          export VAST_HOST=root@1.2.3.4
#          export VAST_PORT=12345
#          export VAST_KEY=~/.ssh/id_ed25519   # optional, defaults to your default key
#   3. (Optional) put your Drive file IDs in ~/.vastrc:
#          H5_DRIVE_ID=...
#          CKPT_DRIVE_ID=...
#
# Commands:
#   vast.sh setup        - one-time remote bootstrap (clones repo, installs deps, gdown H5+ckpt)
#   vast.sh push         - rsync local code → remote /workspace/FMODetect-v2/
#   vast.sh pull         - rsync remote /workspace/experiments/ → local experiments/
#   vast.sh ssh          - open an interactive shell on the box
#   vast.sh run "<cmd>"  - run a command on the box (one-shot)
#   vast.sh train [args] - run scripts/train.py on the box with --resume preset
#   vast.sh infer <img>  - run inference on a local image (uploads, runs, downloads result)
#   vast.sh tail         - tail the last training log on the box

set -euo pipefail

# ---- Load env ----
if [[ -f "$HOME/.vastrc" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.vastrc"
fi
: "${VAST_HOST:?Set VAST_HOST=root@<ip> in env or ~/.vastrc}"
: "${VAST_PORT:?Set VAST_PORT=<port> in env or ~/.vastrc}"
VAST_KEY="${VAST_KEY:-$HOME/.ssh/id_ed25519}"

SSH_OPTS="-i ${VAST_KEY} -p ${VAST_PORT} -o StrictHostKeyChecking=accept-new"
RSYNC_OPTS="-az --info=progress2 --partial -e \"ssh ${SSH_OPTS}\""

REMOTE_ROOT=/workspace/FMODetect-v2
REMOTE_VENV=/workspace/venv
REMOTE_PY=${REMOTE_VENV}/bin/python
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cmd_ssh() {
    exec ssh ${SSH_OPTS} "${VAST_HOST}"
}

cmd_run() {
    local script="$1"
    # cd into the repo if it exists; otherwise run in $HOME so the cmd still works pre-setup
    ssh ${SSH_OPTS} "${VAST_HOST}" "set -e; [ -d ${REMOTE_ROOT} ] && cd ${REMOTE_ROOT}; ${script}"
}

cmd_push() {
    # Push code only — exclude datasets, experiments, venv, etc.
    echo "==> rsync local → ${VAST_HOST}:${REMOTE_ROOT}"
    rsync -az --info=progress2 --partial \
        --delete-excluded \
        --exclude='.venv/' \
        --exclude='venv/' \
        --exclude='datasets/' \
        --exclude='experiments/' \
        --exclude='node_modules/' \
        --exclude='ui/.next/' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.git' \
        --exclude='.mcp.json' \
        --exclude='.env' \
        --exclude='FMODetect-master/' \
        --exclude='docs/' \
        -e "ssh ${SSH_OPTS}" \
        "${LOCAL_ROOT}/" "${VAST_HOST}:${REMOTE_ROOT}/"
}

cmd_pull() {
    echo "==> rsync ${VAST_HOST}:/workspace/experiments/ → ${LOCAL_ROOT}/experiments/"
    mkdir -p "${LOCAL_ROOT}/experiments"
    rsync -az --info=progress2 --partial \
        -e "ssh ${SSH_OPTS}" \
        "${VAST_HOST}:/workspace/experiments/" "${LOCAL_ROOT}/experiments/"
}

cmd_setup() {
    : "${H5_DRIVE_ID:?Set H5_DRIVE_ID in env or ~/.vastrc}"
    : "${CKPT_DRIVE_ID:?Set CKPT_DRIVE_ID in env or ~/.vastrc}"
    echo "==> one-time setup: apt + venv + torch + clone + gdown"
    ssh ${SSH_OPTS} "${VAST_HOST}" bash <<EOF
        set -e

        echo "[1/6] apt deps"
        apt-get update -qq >/dev/null 2>&1
        apt-get install -y -qq git curl unzip ffmpeg rsync python3-venv python3-pip >/dev/null 2>&1

        echo "[2/6] python venv at ${REMOTE_VENV}"
        if [ ! -x ${REMOTE_VENV}/bin/python ]; then
            python3 -m venv ${REMOTE_VENV}
        fi
        ${REMOTE_VENV}/bin/pip install -q --upgrade pip wheel setuptools

        echo "[3/6] pip deps (inside venv — no PEP 668 / debian conflicts)"
        ${REMOTE_VENV}/bin/pip install -q \
            gdown scikit-image h5py opencv-python-headless scipy tqdm \
            "imageio[ffmpeg]" mlflow clearml pydantic pyyaml typer rich tensorboard

        echo "[4/6] torch + torchvision (cu128 wheels for Blackwell sm_120)"
        ${REMOTE_VENV}/bin/pip install -q --index-url https://download.pytorch.org/whl/cu128 \
            torch torchvision

        echo "[5/6] clone repo"
        mkdir -p /workspace/data /workspace/experiments
        if [ ! -d ${REMOTE_ROOT}/.git ]; then
            git clone --depth 1 https://github.com/jai-krishna-0921/FMODetect-v2.git ${REMOTE_ROOT}
        else
            git -C ${REMOTE_ROOT} pull --ff-only || true
        fi

        echo "[6/6] download dataset + checkpoint"
        if [ ! -f /workspace/data/vot_fmo.h5 ]; then
            echo "  gdown vot_fmo.h5 (~16 GB)..."
            ${REMOTE_VENV}/bin/gdown ${H5_DRIVE_ID} -O /workspace/data/vot_fmo.h5
        else
            echo "  vot_fmo.h5 already present"
        fi
        if [ ! -f /workspace/experiments/resume_best.pt ]; then
            echo "  gdown best.pt..."
            ${REMOTE_VENV}/bin/gdown ${CKPT_DRIVE_ID} -O /workspace/experiments/resume_best.pt
        else
            echo "  best.pt already present"
        fi

        echo ""
        echo "==> Summary"
        ls -lh /workspace/data/vot_fmo.h5 /workspace/experiments/resume_best.pt
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        ${REMOTE_VENV}/bin/python -c "import torch; print('torch:', torch.__version__, ' cuda:', torch.cuda.is_available(), ' device:', torch.cuda.get_device_name(0))"
EOF
}

cmd_setup_v2() {
    # From-scratch workflow: VOT2016 + v2 synth + retrain (no resume).
    # Generates the new augmented H5 on the box.
    echo "==> v2 setup: apt + venv + torch + clone + VOT + v2 H5"
    ssh ${SSH_OPTS} "${VAST_HOST}" bash <<EOF
        set -e

        echo "[1/8] apt deps"
        apt-get update -qq >/dev/null 2>&1
        apt-get install -y -qq git curl unzip ffmpeg rsync python3-venv python3-pip >/dev/null 2>&1

        echo "[2/8] python venv at ${REMOTE_VENV}"
        if [ ! -x ${REMOTE_VENV}/bin/python ]; then
            python3 -m venv ${REMOTE_VENV}
        fi
        ${REMOTE_VENV}/bin/pip install -q --upgrade pip wheel setuptools

        echo "[3/8] pip deps"
        ${REMOTE_VENV}/bin/pip install -q \
            scikit-image h5py opencv-python-headless scipy tqdm \
            "imageio[ffmpeg]" mlflow clearml pydantic pyyaml typer rich tensorboard

        echo "[4/8] torch + torchvision (cu128 for Blackwell)"
        ${REMOTE_VENV}/bin/pip install -q --index-url https://download.pytorch.org/whl/cu128 \
            torch torchvision

        echo "[5/8] clone repo"
        mkdir -p /workspace/data /workspace/experiments
        if [ ! -d ${REMOTE_ROOT}/.git ]; then
            git clone --depth 1 https://github.com/jai-krishna-0921/FMODetect-v2.git ${REMOTE_ROOT}
        else
            git -C ${REMOTE_ROOT} pull --ff-only || true
        fi

        echo "[6/8] download VOT2016 backgrounds (~1.4 GB)"
        if [ ! -f /workspace/data/vot2016/.done ]; then
            cd ${REMOTE_ROOT}
            PYTHONPATH=. ${REMOTE_VENV}/bin/python scripts/download_vot2016_only.py \
                --out /workspace/data/vot2016
        else
            echo "  VOT2016 already present"
        fi

        echo "[7/8] generate v2 synthetic H5 (~30 min on Vast CPU)"
        if [ ! -f /workspace/data/vot_fmo_v2.h5 ]; then
            cd ${REMOTE_ROOT}
            PYTHONPATH=. ${REMOTE_VENV}/bin/python -m src.fmodetect.data.build_dataset \
                --bg /workspace/data/vot2016 \
                --patterns /workspace/data/patterns \
                --out /workspace/data/vot_fmo_v2.h5 \
                --n 5000 --shape 256 512 --seed 42
        else
            echo "  vot_fmo_v2.h5 already present"
        fi

        echo "[8/8] verify"
        ls -lh /workspace/data/vot_fmo_v2.h5
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        ${REMOTE_VENV}/bin/python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
EOF
}

cmd_train_v2() {
    # From-scratch training with the v2 config + new H5.
    local epochs="${RESUME_TOTAL_EPOCHS:-120}"
    echo "==> remote train v2 (from scratch → epoch ${epochs})"
    ssh ${SSH_OPTS} "${VAST_HOST}" "
        set -e
        cd ${REMOTE_ROOT}
        PYTHONPATH=. PYTHONUNBUFFERED=1 ${REMOTE_PY} -u scripts/train.py \\
            --config configs/vast_rerun.yaml \\
            --epochs ${epochs}
    "
}

cmd_train() {
    # Default: resume from epoch 29 ckpt, train to epoch 45
    local epochs="${RESUME_TOTAL_EPOCHS:-45}"
    echo "==> remote train (resume → epoch ${epochs})"
    ssh ${SSH_OPTS} "${VAST_HOST}" "
        set -e
        cd ${REMOTE_ROOT}
        PYTHONPATH=. PYTHONUNBUFFERED=1 ${REMOTE_PY} -u scripts/train.py \\
            --config configs/vast.yaml \\
            --resume /workspace/experiments/resume_best.pt \\
            --epochs ${epochs} \\
            $*
    "
}

cmd_infer() {
    local img="$1"
    local bgr="${2:-}"
    local stem
    stem="$(basename "${img%.*}")"
    local remote_img=/workspace/_infer/${stem}.png
    local remote_bgr=/workspace/_infer/${stem}_bgr.png

    ssh ${SSH_OPTS} "${VAST_HOST}" "mkdir -p /workspace/_infer"
    scp ${SSH_OPTS} "${img}" "${VAST_HOST}:${remote_img}"
    if [[ -n "$bgr" ]]; then
        scp ${SSH_OPTS} "${bgr}" "${VAST_HOST}:${remote_bgr}"
        bgr_arg="--bgr ${remote_bgr}"
    else
        bgr_arg=""
    fi

    ssh ${SSH_OPTS} "${VAST_HOST}" "
        set -e
        cd ${REMOTE_ROOT}
        CKPT=\$(ls -t /workspace/experiments/checkpoints/run_*/best.pt 2>/dev/null | head -1)
        [ -z \"\$CKPT\" ] && CKPT=/workspace/experiments/resume_best.pt
        echo Using ckpt: \$CKPT
        PYTHONPATH=. ${REMOTE_PY} scripts/infer.py --ckpt \$CKPT \\
            --image ${remote_img} ${bgr_arg} \\
            --out /workspace/_infer/${stem}_out.png
    "
    mkdir -p "${LOCAL_ROOT}/experiments/infer_remote"
    scp ${SSH_OPTS} "${VAST_HOST}:/workspace/_infer/${stem}_out.png" \
        "${LOCAL_ROOT}/experiments/infer_remote/${stem}_out.png" || true
    scp ${SSH_OPTS} "${VAST_HOST}:/workspace/_infer/${stem}_out.tdf.npy" \
        "${LOCAL_ROOT}/experiments/infer_remote/${stem}_out.tdf.npy" || true
    echo "Pulled → experiments/infer_remote/${stem}_out.png"
}

cmd_tail() {
    ssh ${SSH_OPTS} "${VAST_HOST}" "
        latest=\$(ls -td /workspace/experiments/tb/run_* 2>/dev/null | head -1)
        if [ -z \"\$latest\" ]; then
            echo 'no training runs found on remote yet'
            exit 0
        fi
        echo 'latest run dir:' \$latest
        ls /workspace/experiments/checkpoints/\$(basename \$latest)/ 2>/dev/null || true
    "
}

case "${1:-}" in
    setup)     cmd_setup ;;
    setup_v2)  cmd_setup_v2 ;;
    push)      cmd_push ;;
    pull)      cmd_pull ;;
    ssh)       cmd_ssh ;;
    run)       shift; cmd_run "$*" ;;
    train)     shift; cmd_train "$@" ;;
    train_v2)  shift; cmd_train_v2 "$@" ;;
    infer)     shift; cmd_infer "$@" ;;
    tail)      cmd_tail ;;
    *)
        echo "Usage: $0 {setup|setup_v2|push|pull|ssh|run|train|train_v2|infer|tail}"
        echo "Commands:"
        echo "  setup            v1 bootstrap (gdown 16 GB H5 + best.pt)"
        echo "  setup_v2         v2 bootstrap (download VOT + generate v2 H5 on box)"
        echo "  push             rsync local code → remote"
        echo "  pull             rsync remote experiments/ → local"
        echo "  ssh              interactive shell"
        echo "  run \"<cmd>\"     one-shot remote command"
        echo "  train [args]     v1 resume training (epoch 29 → 45)"
        echo "  train_v2         v2 from-scratch with augmented synth + tuned loss"
        echo "  infer <img>      one-shot inference on a local image"
        echo "  tail             show latest run dir + checkpoints"
        exit 2
        ;;
esac
