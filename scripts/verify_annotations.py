"""Verify dataset annotation completeness.

Walks every downloaded dataset directory and reports:
  - VOT2016: per-sequence (frames_present, ann_present, gt_lines, first_gt)
  - falling / TbD / TbD-3D: per-sequence (n_frames, n_gt_masks, sample GT bbox)

Outputs a JSON report and prints a colourised summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.fmodetect.eval.datasets import load_eval_dataset


def check_vot(root: Path) -> dict:
    seqs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        frames = sorted(d.glob("*.jpg"))
        gt = d / "groundtruth.txt"
        gt_lines = 0
        first_gt = None
        if gt.exists():
            with gt.open() as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            gt_lines = len(lines)
            if lines:
                first_gt = lines[0]
        seqs.append({
            "name": d.name,
            "n_frames": len(frames),
            "has_groundtruth": gt.exists(),
            "gt_lines": gt_lines,
            "frames_eq_gt": gt_lines == len(frames),
            "first_gt": first_gt,
        })
    n_seqs = len(seqs)
    n_with_ann = sum(1 for s in seqs if s["has_groundtruth"])
    return {
        "type": "vot2016",
        "root": str(root),
        "n_sequences": n_seqs,
        "n_with_annotations": n_with_ann,
        "all_annotated": n_with_ann == n_seqs and n_seqs > 0,
        "sequences": seqs,
    }


def check_eval_dataset(root: Path, name: str) -> dict:
    if not root.exists():
        return {"type": name, "root": str(root), "exists": False}
    seqs_dict = load_eval_dataset(root)
    if not seqs_dict:
        return {"type": name, "root": str(root), "exists": True, "n_sequences": 0,
                "note": "no recognizable sequences"}
    out_seqs = []
    for seq_name, frames in seqs_dict.items():
        n_gt = sum(1 for f in frames if f.gt_traj_yx is not None)
        # sample a gt point for sanity
        sample = next((f for f in frames if f.gt_traj_yx is not None), None)
        s_info = None
        if sample is not None:
            s_info = {
                "sample_gt_radius_px": sample.gt_radius_px,
                "sample_gt_npoints": len(sample.gt_traj_yx),
                "sample_gt_bbox_yxyx": list(sample.gt_bbox_yxyx),
                "sample_image": str(sample.image_path),
            }
        out_seqs.append({
            "name": seq_name,
            "n_frames": len(frames),
            "n_with_gt": n_gt,
            "gt_coverage": n_gt / max(1, len(frames)),
            "background_present": any(f.background_path for f in frames),
            "sample": s_info,
        })
    return {
        "type": name,
        "root": str(root),
        "exists": True,
        "n_sequences": len(seqs_dict),
        "n_frames_total": sum(len(v) for v in seqs_dict.values()),
        "n_with_gt_total": sum(s["n_with_gt"] for s in out_seqs),
        "sequences": out_seqs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    p.add_argument("--out", type=Path, default=Path("datasets/_annotation_report.json"))
    args = p.parse_args()

    report = {}
    vot_root = args.datasets_root / "vot2016"
    if vot_root.exists():
        report["vot2016"] = check_vot(vot_root)
    for name in ("falling", "TbD-3D", "TbD"):
        report[name] = check_eval_dataset(args.datasets_root / "eval" / name, name)
    args.out.write_text(json.dumps(report, indent=2))

    # Print summary
    print(f"\nAnnotation report written to {args.out}\n")
    for k, v in report.items():
        if not v.get("exists", True):
            print(f"  [{k:10s}] not downloaded yet")
            continue
        if k == "vot2016":
            print(f"  [{k:10s}] {v['n_sequences']} sequences, "
                  f"{v['n_with_annotations']}/{v['n_sequences']} annotated  "
                  f"{'OK' if v['all_annotated'] else 'INCOMPLETE'}")
        else:
            n_seq = v.get("n_sequences", 0)
            print(f"  [{k:10s}] {n_seq} sequences, "
                  f"{v.get('n_frames_total', 0)} frames, "
                  f"{v.get('n_with_gt_total', 0)} with GT")
    print()


if __name__ == "__main__":
    main()
