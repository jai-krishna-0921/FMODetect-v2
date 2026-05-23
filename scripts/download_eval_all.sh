#!/usr/bin/env bash
# Download every eval dataset we need + the original FMO (2017) dataset.
# Resumes partial downloads via curl -C -. Idempotent: re-runs skip already-done dirs.
#
# Datasets:
#   VOT2016                - backgrounds + per-frame 4-point polygon GT (for training synth)
#   falling_objects        - 6 sequences, per-frame bbox GT "x y w h"
#   TbD-3D                 - 3D-rotating spherical objects, high-speed GT
#   TbD                    - 25 GB main eval, high-speed sub-frame trajectory GT
#   FMO 2017               - 16 sequences from Rozumnyi 2017 + V1/V2 GT (text format)
#
# Usage:
#   bash scripts/download_eval_all.sh                # download all
#   bash scripts/download_eval_all.sh tbd            # one target
#
# Logs to datasets/_logs/<name>.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/datasets"
LOGDIR="$DATA/_logs"
mkdir -p "$DATA"/{vot2016,eval/falling,eval/TbD-3D,eval/TbD,eval/FMO} "$LOGDIR"

PTAK="https://ptak.felk.cvut.cz/personal/rozumden"
CMP="http://cmp.felk.cvut.cz/fmo/files"

CURL="curl -kL -C - --retry 6 --retry-delay 10 --retry-all-errors --connect-timeout 30"

# ---- helpers ----

fetch_zip() {
    # $1 = url, $2 = output zip, $3 = unzip dest, $4 = .done flag
    local url="$1" zf="$2" dest="$3" flag="$4"
    if [[ -f "$flag" ]]; then echo "[$(basename "$dest")] already done"; return 0; fi
    mkdir -p "$dest"
    echo "[$(basename "$dest")] downloading $url"
    $CURL "$url" -o "$zf"
    echo "[$(basename "$dest")] unzipping"
    unzip -q -o "$zf" -d "$dest"
    rm -f "$zf"
    touch "$flag"
    echo "[$(basename "$dest")] OK"
}

# ---- targets ----

fetch_falling() {
    fetch_zip "$PTAK/falling_imgs_gt.zip" \
              "$DATA/eval/falling/falling_imgs_gt.zip" \
              "$DATA/eval/falling" "$DATA/eval/falling/.done"
}

fetch_tbd3d() {
    fetch_zip "$PTAK/TbD-3D-imgs.zip" \
              "$DATA/eval/TbD-3D/TbD-3D-imgs.zip" \
              "$DATA/eval/TbD-3D" "$DATA/eval/TbD-3D/.done"
}

fetch_tbd() {
    fetch_zip "$PTAK/TbD_imgs_upd.zip" \
              "$DATA/eval/TbD/TbD_imgs_upd.zip" \
              "$DATA/eval/TbD" "$DATA/eval/TbD/.done"
}

fetch_fmo() {
    local d="$DATA/eval/FMO"
    if [[ -f "$d/.done" ]]; then echo "[FMO] already done"; return 0; fi
    echo "[FMO] downloading experiment zip (194 MB)"
    $CURL "$CMP/fmo-cpp-experiment-2017-05-26.zip" -o "$d/experiment.zip"
    echo "[FMO] downloading GT V1 (text)"
    $CURL "$CMP/gt-fmo-txt-2017-05-26.zip" -o "$d/gt_v1.zip"
    echo "[FMO] downloading GT V2 (text)"
    $CURL "$CMP/gt-fmov2-txt-2017-05-26.zip" -o "$d/gt_v2.zip"
    echo "[FMO] unzipping"
    unzip -q -o "$d/experiment.zip" -d "$d"
    unzip -q -o "$d/gt_v1.zip" -d "$d/gt_v1"
    unzip -q -o "$d/gt_v2.zip" -d "$d/gt_v2"
    rm -f "$d/experiment.zip" "$d/gt_v1.zip" "$d/gt_v2.zip"
    touch "$d/.done"
    echo "[FMO] OK"
}

# Run targets
case "${1:-all}" in
    falling) fetch_falling 2>&1 | tee "$LOGDIR/falling.log" ;;
    tbd3d)   fetch_tbd3d   2>&1 | tee "$LOGDIR/tbd3d.log" ;;
    tbd)     fetch_tbd     2>&1 | tee "$LOGDIR/tbd.log" ;;
    fmo)     fetch_fmo     2>&1 | tee "$LOGDIR/fmo.log" ;;
    all)
        # Already-done ones are no-ops. TbD goes last because it's huge.
        fetch_falling 2>&1 | tee "$LOGDIR/falling.log"
        fetch_tbd3d   2>&1 | tee "$LOGDIR/tbd3d.log"
        fetch_fmo     2>&1 | tee "$LOGDIR/fmo.log"
        fetch_tbd     2>&1 | tee "$LOGDIR/tbd.log"
        ;;
    *) echo "Usage: $0 {all|falling|tbd3d|tbd|fmo}" ; exit 2 ;;
esac

echo ""
echo "==> final state"
for d in falling TbD-3D TbD FMO; do
    p="$DATA/eval/$d"
    sz=$(du -sh "$p" 2>/dev/null | cut -f1)
    [[ -f "$p/.done" ]] && status="DONE" || status="incomplete"
    echo "  $d: $sz ($status)"
done
