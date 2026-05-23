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
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cmd_ssh() {
    exec ssh ${SSH_OPTS} "${VAST_HOST}"
}

cmd_run() {
    local script="$1"
    ssh ${SSH_OPTS} "${VAST_HOST}" "set -e; cd ${REMOTE_ROOT} && ${script}"
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
    echo "==> one-time setup: apt + pip + clone + gdown"
    ssh ${SSH_OPTS} "${VAST_HOST}" "
        set -e
        apt-get update -qq
        apt-get install -y -qq git curl unzip ffmpeg rsync >/dev/null
        pip install -q --upgrade pip
        pip install -q gdown scikit-image h5py opencv-python-headless scipy tqdm \\
            imageio[ffmpeg] mlflow clearml pydantic pyyaml typer rich tensorboard

        mkdir -p /workspace/data /workspace/experiments
        if [ ! -d ${REMOTE_ROOT}/.git ]; then
            git clone --depth 1 https://github.com/jai-krishna-0921/FMODetect-v2.git ${REMOTE_ROOT}
        fi

        if [ ! -f /workspace/data/vot_fmo.h5 ]; then
            echo '==> gdown vot_fmo.h5 (~16 GB)'
            gdown --id ${H5_DRIVE_ID} -O /workspace/data/vot_fmo.h5
        fi
        if [ ! -f /workspace/experiments/resume_best.pt ]; then
            echo '==> gdown best.pt'
            gdown --id ${CKPT_DRIVE_ID} -O /workspace/experiments/resume_best.pt
        fi
        ls -lh /workspace/data/vot_fmo.h5 /workspace/experiments/resume_best.pt
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    "
}

cmd_train() {
    # Default: resume from epoch 29 ckpt, train to epoch 45
    local epochs="${RESUME_TOTAL_EPOCHS:-45}"
    echo "==> remote train (resume → epoch ${epochs})"
    ssh ${SSH_OPTS} "${VAST_HOST}" "
        set -e
        cd ${REMOTE_ROOT}
        PYTHONPATH=. PYTHONUNBUFFERED=1 python -u scripts/train.py \\
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
        PYTHONPATH=. python scripts/infer.py --ckpt \$CKPT \\
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
    setup)  cmd_setup ;;
    push)   cmd_push ;;
    pull)   cmd_pull ;;
    ssh)    cmd_ssh ;;
    run)    shift; cmd_run "$*" ;;
    train)  shift; cmd_train "$@" ;;
    infer)  shift; cmd_infer "$@" ;;
    tail)   cmd_tail ;;
    *)
        echo "Usage: $0 {setup|push|pull|ssh|run|train|infer|tail}"
        echo "Commands:"
        echo "  setup            one-time remote bootstrap (clone repo + gdown H5+ckpt)"
        echo "  push             rsync local code → remote"
        echo "  pull             rsync remote experiments/ → local"
        echo "  ssh              interactive shell"
        echo "  run \"<cmd>\"     one-shot remote command"
        echo "  train [args]     resume training (default: epoch 29 → 45)"
        echo "  infer <img>      one-shot inference on a local image, pulls result back"
        echo "  tail             show latest run dir + its checkpoints"
        exit 2
        ;;
esac
