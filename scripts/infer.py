"""Inference CLI:
    .venv/bin/python scripts/infer.py --image X.png --bgr Y.png --ckpt experiments/.../best.pt
    .venv/bin/python scripts/infer.py --video clip.mp4 --ckpt experiments/.../best.pt --out detections.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.fmodetect.inference.runner import infer_pair, infer_video, load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--image", type=Path)
    sub.add_argument("--video", type=Path)
    p.add_argument("--bgr", type=Path)
    p.add_argument("--out", type=Path, default=Path("detections.mp4"))
    args = p.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    if args.image is not None:
        if args.bgr is None:
            raise SystemExit("--bgr required with --image")
        img = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bg = cv2.cvtColor(cv2.imread(str(args.bgr)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        res = infer_pair(model, img, bg, device=device)
        cv2.imwrite(str(args.out.with_suffix(".png")), res.overlay)
        np.save(args.out.with_suffix(".tdf.npy"), res.tdf)
        print(f"wrote {args.out.with_suffix('.png')}")
    else:
        info = infer_video(model, args.video, args.out, device=device)
        print(info)


if __name__ == "__main__":
    main()
