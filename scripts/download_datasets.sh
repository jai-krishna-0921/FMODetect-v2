#!/usr/bin/env bash
# Download all FMO datasets needed for FMODetect-v2.
# Logs to datasets/_logs/. Use `tail -f datasets/_logs/<name>.log` to monitor.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/datasets"
LOGDIR="$DATA/_logs"
mkdir -p "$DATA"/{vot2016,eval/falling,eval/TbD-3D,eval/TbD} "$LOGDIR"

# Curl flags: -k (self-signed CVUT cert), -L (follow redirects), -C - (resume), --retry
CURL="curl -kL -C - --retry 5 --retry-delay 10 --retry-all-errors --connect-timeout 30"

PTAK_BASE="https://ptak.felk.cvut.cz/personal/rozumden"
VOT_BASE="https://data.votchallenge.net/vot2016/main"

# ---- falling_objects ----
fetch_falling() {
    local f="$DATA/eval/falling/falling_imgs_gt.zip"
    [[ -f "$DATA/eval/falling/.done" ]] && { echo "[falling] already done"; return 0; }
    echo "[falling] downloading ~1.0 GB..."
    $CURL "$PTAK_BASE/falling_imgs_gt.zip" -o "$f"
    echo "[falling] unzipping..."
    unzip -q -o "$f" -d "$DATA/eval/falling/" && rm "$f" && touch "$DATA/eval/falling/.done"
    echo "[falling] OK"
}

# ---- TbD-3D ----
fetch_tbd3d() {
    local f="$DATA/eval/TbD-3D/TbD-3D-imgs.zip"
    [[ -f "$DATA/eval/TbD-3D/.done" ]] && { echo "[TbD-3D] already done"; return 0; }
    echo "[TbD-3D] downloading ~2.7 GB..."
    $CURL "$PTAK_BASE/TbD-3D-imgs.zip" -o "$f"
    echo "[TbD-3D] unzipping..."
    unzip -q -o "$f" -d "$DATA/eval/TbD-3D/" && rm "$f" && touch "$DATA/eval/TbD-3D/.done"
    echo "[TbD-3D] OK"
}

# ---- TbD (LARGE 25 GB) ----
fetch_tbd() {
    local f="$DATA/eval/TbD/TbD_imgs_upd.zip"
    [[ -f "$DATA/eval/TbD/.done" ]] && { echo "[TbD] already done"; return 0; }
    echo "[TbD] downloading ~25.2 GB (this will take a while)..."
    $CURL "$PTAK_BASE/TbD_imgs_upd.zip" -o "$f"
    echo "[TbD] unzipping..."
    unzip -q -o "$f" -d "$DATA/eval/TbD/" && rm "$f" && touch "$DATA/eval/TbD/.done"
    echo "[TbD] OK"
}

# ---- VOT2016 ----
# data.votchallenge.net/vot2016/main/description.json lists 60 sequences.
# Each sequence has a colour.zip with JPEGs.
fetch_vot2016() {
    [[ -f "$DATA/vot2016/.done" ]] && { echo "[vot2016] already done"; return 0; }
    echo "[vot2016] fetching description.json..."
    $CURL "$VOT_BASE/description.json" -o "$DATA/vot2016/description.json"
    local count
    count=$(python3 -c "import json; print(len(json.load(open('$DATA/vot2016/description.json'))['sequences']))")
    echo "[vot2016] $count sequences to download"
    python3 -c "
import json, os, subprocess, sys
desc = json.load(open('$DATA/vot2016/description.json'))
seqs = desc['sequences']
base = '$VOT_BASE'
out = '$DATA/vot2016'
def fetch(url, dst):
    subprocess.run(['curl','-kL','-C','-','--retry','5','--retry-delay','10','--retry-all-errors','--connect-timeout','30', url, '-o', dst], check=True)
for i, s in enumerate(seqs):
    name = s['name']
    seq_dir = os.path.join(out, name)
    done = os.path.join(seq_dir, '.done')
    ann_done = os.path.join(seq_dir, '.ann_done')
    os.makedirs(seq_dir, exist_ok=True)

    # Frames (color.zip)
    if not os.path.exists(done):
        ch = s['channels']['color']
        url = base + '/' + ch['url']
        zf = os.path.join(seq_dir, 'color.zip')
        print(f'[vot2016 {i+1}/{len(seqs)} frames ] {name}')
        fetch(url, zf)
        subprocess.run(['unzip','-q','-o', zf, '-d', seq_dir], check=True)
        os.remove(zf)
        open(done,'w').close()

    # Annotations (<name>.zip)
    if not os.path.exists(ann_done):
        ann_url = base + '/' + s['annotations']['url']
        azf = os.path.join(seq_dir, 'annotations.zip')
        print(f'[vot2016 {i+1}/{len(seqs)} ann    ] {name}')
        fetch(ann_url, azf)
        subprocess.run(['unzip','-q','-o', azf, '-d', seq_dir], check=True)
        os.remove(azf)
        open(ann_done,'w').close()
"
    touch "$DATA/vot2016/.done"
    echo "[vot2016] OK"
}

# Run requested target(s)
case "${1:-all}" in
    falling)  fetch_falling   2>&1 | tee "$LOGDIR/falling.log"   ;;
    tbd3d)    fetch_tbd3d     2>&1 | tee "$LOGDIR/tbd3d.log"     ;;
    tbd)      fetch_tbd       2>&1 | tee "$LOGDIR/tbd.log"       ;;
    vot2016)  fetch_vot2016   2>&1 | tee "$LOGDIR/vot2016.log"   ;;
    all)
        # Run small ones first to fail fast on issues; TbD last since it's huge.
        fetch_vot2016  2>&1 | tee "$LOGDIR/vot2016.log"
        fetch_falling  2>&1 | tee "$LOGDIR/falling.log"
        fetch_tbd3d    2>&1 | tee "$LOGDIR/tbd3d.log"
        fetch_tbd      2>&1 | tee "$LOGDIR/tbd.log"
        ;;
    *) echo "Usage: $0 {all|vot2016|falling|tbd3d|tbd}" ; exit 2 ;;
esac
