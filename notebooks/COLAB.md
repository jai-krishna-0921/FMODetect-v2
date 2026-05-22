# Colab training notebook

The Colab notebook for training FMODetect-v2 lives at:

> https://colab.research.google.com/ (paste your own copy)

Cell-by-cell plan (see chat transcript or `scripts/eval.py`/`scripts/train.py` for what each cell calls):

1. **GPU check** — verifies `nvidia-smi` + `torch.cuda.is_available()`.
2. **Mount Drive** — `/content/drive/MyDrive/FMODetect-v2`.
3. **Clone** — `git clone https://github.com/jai-krishna-0921/FMODetect-v2.git`.
4. **Install deps** — `scikit-image, h5py, mlflow, fastapi, ...` (torch already on Colab).
5. **Download VOT2016** — 60 sequences + per-sequence annotations to Drive.
6. **Download eval sets in background** — falling (1 GB), TbD-3D (2.7 GB), TbD (25 GB) via `subprocess.Popen`.
7. **Verify annotations** — runs `scripts/verify_annotations.py`.
8. **Generate synth H5** — `python -m src.fmodetect.data.build_dataset --n 5000`.
9. **Write `configs/colab.yaml`** — bs 16, 40 epochs, AMP off, Drive paths.
10. **Train** — `python scripts/train.py --config configs/colab.yaml`.
11. **Eval** — waits for eval downloads, runs `scripts/eval.py` on each ready dataset.
12. **Smoke** — runs inference on the original FMODetect example image to visually verify.

If you re-run a cell, downloads/datasets/H5 are short-circuited by `.done` flags and existence checks — nothing redoes work unnecessarily.

