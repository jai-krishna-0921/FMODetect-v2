"""Fetch VOT2016 per-sequence annotations (groundtruth.txt etc.).

Independent of the main download script — safe to run alongside it.
Skips sequences that already have .ann_done.

Usage: PYTHONPATH=. .venv/bin/python scripts/fetch_vot_annotations.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/vot2016"))
    p.add_argument("--base", default="https://data.votchallenge.net/vot2016/main")
    args = p.parse_args()

    desc_path = args.root / "description.json"
    if not desc_path.exists():
        raise SystemExit(f"{desc_path} missing; run download_datasets.sh first")
    desc = json.loads(desc_path.read_text())

    n_ok = n_skip = 0
    for i, s in enumerate(desc["sequences"]):
        name = s["name"]
        seq_dir = args.root / name
        ann_done = seq_dir / ".ann_done"
        if ann_done.exists():
            n_skip += 1
            continue
        seq_dir.mkdir(parents=True, exist_ok=True)
        url = f"{args.base}/{s['annotations']['url']}"
        zf = seq_dir / "annotations.zip"
        print(f"[ann {i+1}/{len(desc['sequences'])}] {name}")
        try:
            subprocess.run(
                ["curl", "-kL", "-C", "-", "--retry", "5", "--retry-delay", "5",
                 "--retry-all-errors", "--connect-timeout", "30",
                 "-sS",  # silent but show errors
                 url, "-o", str(zf)],
                check=True,
            )
            subprocess.run(["unzip", "-q", "-o", str(zf), "-d", str(seq_dir)], check=True)
            zf.unlink()
            ann_done.write_text("")
            n_ok += 1
        except subprocess.CalledProcessError as e:
            print(f"  ! failed for {name}: {e}")
    print(f"\n{n_ok} fetched, {n_skip} already done")


if __name__ == "__main__":
    main()
