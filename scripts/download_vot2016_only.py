"""Download VOT2016 frames + per-sequence annotations to a target dir.

Used by `vast.sh setup_v2` to bootstrap the dataset on a fresh Vast box.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


VOT_BASE = "https://data.votchallenge.net/vot2016/main"


def fetch(url: str, dst: Path) -> None:
    subprocess.check_call([
        "curl", "-kL", "-C", "-", "--retry", "5", "--retry-delay", "5",
        "--retry-all-errors", "--connect-timeout", "30", "-sS",
        url, "-o", str(dst),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=VOT_BASE)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    desc_path = args.out / "description.json"
    if not desc_path.exists():
        print("Fetching description.json...")
        fetch(f"{args.base}/description.json", desc_path)
    desc = json.loads(desc_path.read_text())
    seqs = desc["sequences"]
    print(f"{len(seqs)} sequences to fetch")

    for i, s in enumerate(seqs):
        name = s["name"]
        seq_dir = args.out / name
        seq_dir.mkdir(exist_ok=True)
        done = seq_dir / ".done"
        ann_done = seq_dir / ".ann_done"

        if not done.exists():
            url = f"{args.base}/{s['channels']['color']['url']}"
            zf = seq_dir / "color.zip"
            print(f"[{i+1}/{len(seqs)} frames] {name}")
            fetch(url, zf)
            subprocess.check_call(["unzip", "-q", "-o", str(zf), "-d", str(seq_dir)])
            zf.unlink()
            done.write_text("")

        if not ann_done.exists():
            url = f"{args.base}/{s['annotations']['url']}"
            zf = seq_dir / "annotations.zip"
            print(f"[{i+1}/{len(seqs)} ann   ] {name}")
            fetch(url, zf)
            subprocess.check_call(["unzip", "-q", "-o", str(zf), "-d", str(seq_dir)])
            zf.unlink()
            ann_done.write_text("")

    (args.out / ".done").touch()
    n_frames = sum(1 for _ in args.out.rglob("*.jpg"))
    print(f"\nVOT2016 OK: {len(seqs)} sequences, {n_frames} frames")


if __name__ == "__main__":
    main()
