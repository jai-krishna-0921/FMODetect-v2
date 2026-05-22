#!/usr/bin/env bash
# Unified status snapshot — datasets, GPU, training, MLflow, recent artifacts.
# Run repeatedly via:  watch -n 10 bash scripts/status.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; NC="\033[0m"
hr() { printf "${CYAN}%s${NC}\n" "──────────────────────────────────────────────────────────────"; }
say() { printf "${BOLD}%s${NC} %s\n" "$1" "${2:-}"; }
ok() { printf "  ${GREEN}✓${NC} %s\n" "$1"; }
warn() { printf "  ${YELLOW}!${NC} %s\n" "$1"; }
miss() { printf "  ${RED}✗${NC} %s\n" "$1"; }

printf "\n${BOLD}FMODetect-v2  status @ %s${NC}\n" "$(date '+%H:%M:%S')"
hr

# --- GPU ---
say "GPU"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader | sed 's/^/  /'
else
    miss "nvidia-smi not found"
fi
hr

# --- Datasets ---
say "Datasets"
VOT_DONE=$(ls datasets/vot2016/*/.done 2>/dev/null | wc -l)
VOT_ANN=$(ls datasets/vot2016/*/.ann_done 2>/dev/null | wc -l)
VOT_FRAMES=$(find datasets/vot2016 -name "*.jpg" 2>/dev/null | wc -l)
printf "  vot2016 frames:%s  ann:%s/60  total frames:%s\n" "$VOT_DONE" "$VOT_ANN" "$VOT_FRAMES"
for ds in falling TbD-3D TbD; do
    if [ -f "datasets/eval/$ds/.done" ]; then
        sz=$(du -sh "datasets/eval/$ds" 2>/dev/null | awk '{print $1}')
        nseq=$(find "datasets/eval/$ds" -maxdepth 2 -mindepth 1 -type d 2>/dev/null | wc -l)
        ok "$ds  size:$sz  ~seqs:$nseq"
    else
        warn "$ds  pending"
    fi
done
SYNTH=$(ls datasets/synth/*.h5 2>/dev/null)
if [ -n "$SYNTH" ]; then
    for h in $SYNTH; do
        sz=$(du -sh "$h" 2>/dev/null | awk '{print $1}')
        ok "synthetic: $(basename "$h") ($sz)"
    done
else
    warn "no synthetic H5 built yet"
fi
hr

# --- Background processes ---
say "Background"
DL_PID=$(pgrep -fa "download_datasets" | head -1 | awk '{print $1}')
if [ -n "$DL_PID" ]; then
    ok "download running PID=$DL_PID"
    # which curl is active right now?
    CUR=$(pgrep -fa "curl.*ptak\|curl.*votchallenge" | head -1 | grep -oE '[^/]+\.zip' | head -1)
    [ -n "$CUR" ] && printf "       fetching: %s\n" "$CUR"
else
    warn "no download process"
fi
TRAIN_PID=$(pgrep -fa "scripts/train.py" | head -1 | awk '{print $1}')
[ -n "$TRAIN_PID" ] && ok "training running PID=$TRAIN_PID" || warn "no training process"
API_PID=$(pgrep -fa "uvicorn api.main" | head -1 | awk '{print $1}')
[ -n "$API_PID" ] && ok "API server PID=$API_PID" || warn "API not running"
hr

# --- MLflow runs ---
say "MLflow"
if [ -d experiments/mlruns ]; then
    EXP_DIRS=$(find experiments/mlruns -maxdepth 1 -mindepth 1 -type d | wc -l)
    LATEST_RUN=$(find experiments/mlruns -name "meta.yaml" -path "*/run_*/*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
    printf "  experiments: %s\n" "$EXP_DIRS"
    if [ -n "$LATEST_RUN" ]; then
        RUN_DIR=$(dirname "$LATEST_RUN")
        printf "  latest run: %s\n" "$(basename "$RUN_DIR")"
        for m in val/total train/total val/tdf_nll val/matting_bce; do
            MF="$RUN_DIR/metrics/${m//\//_}"
            [ -f "$MF" ] && printf "    %s = %s (last)\n" "$m" "$(tail -1 "$MF" | awk '{print $2}')"
        done
    fi
else
    miss "experiments/mlruns missing"
fi
hr

# --- Checkpoints ---
say "Checkpoints"
CKPTS=$(ls -t experiments/checkpoints/*/best.pt 2>/dev/null | head -3)
if [ -n "$CKPTS" ]; then
    for c in $CKPTS; do
        sz=$(du -h "$c" 2>/dev/null | awk '{print $1}')
        printf "  %s  (%s)\n" "$c" "$sz"
    done
else
    miss "no checkpoints yet"
fi
hr

# --- Disk ---
say "Disk"
df -h "$ROOT" | tail -1 | awk '{printf "  free: %s of %s (%s used)\n", $4, $2, $5}'
hr
printf "\n"
