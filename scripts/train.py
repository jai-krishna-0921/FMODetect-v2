"""CLI entrypoint:
    .venv/bin/python scripts/train.py --config configs/default.yaml
    .venv/bin/python scripts/train.py --config configs/vast.yaml \
        --resume experiments/checkpoints/run_X/best.pt --epochs 45
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.fmodetect.training.loop import train
from src.fmodetect.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--resume", type=Path, default=None,
                   help="Optional checkpoint to resume from (model+opt+scaler+epoch).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override config train.epochs (total target, not delta).")
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg,
          resume_from=str(args.resume) if args.resume else None,
          epochs_override=args.epochs)


if __name__ == "__main__":
    main()
