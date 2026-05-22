"""CLI entrypoint: .venv/bin/python scripts/train.py --config configs/default.yaml"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.fmodetect.training.loop import train
from src.fmodetect.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
