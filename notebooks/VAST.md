# Vast.ai — drive remote training from your local machine

Two workflows. **Pick whichever you like**, they don't conflict.

| | Workflow A: rsync + ssh wrapper | Workflow B: VS Code Remote-SSH |
|---|---|---|
| Editor | any (vim, JetBrains, VS Code, etc.) | VS Code only |
| Sync mechanism | manual `vast.sh push` per change | automatic on save |
| What runs locally | nothing heavy (just rsync/ssh) | nothing (full IDE talks to remote) |
| Best for | one-shot training runs, scripted | iterative dev with breakpoints |

---

## Workflow A — `scripts/vast.sh` (recommended for this project)

### One-time prep (local; meter not running yet)

**1. Make two Drive files shareable** and grab their file IDs:
- `My Drive/FMODetect-v2/datasets/synth/vot_fmo.h5`
- `My Drive/FMODetect-v2/experiments/checkpoints/run_20260522_185205/best.pt`

For each: right-click → Share → "Anyone with the link" → copy URL → file ID is the 33-char chunk between `/d/` and `/view`.

**2. Create `~/.vastrc`** with your credentials and Drive IDs (template below). The script reads this every time it runs.

```bash
cat > ~/.vastrc <<'EOF'
# Vast.ai connection (fill in after you rent — Vast shows these in the instance row)
export VAST_HOST=root@1.2.3.4
export VAST_PORT=12345
export VAST_KEY=~/.ssh/id_ed25519     # the key you added to Vast

# Google Drive file IDs for the H5 dataset and checkpoint
export H5_DRIVE_ID=PASTE_H5_ID_HERE
export CKPT_DRIVE_ID=PASTE_CKPT_ID_HERE

# Optional: how many total epochs to train to (resumes from epoch 29)
export RESUME_TOTAL_EPOCHS=45
EOF
chmod 600 ~/.vastrc
```

### When you rent

1. **Rent** an RTX PRO 6000 (or any 80 GB+) box on Vast. Wait for `RUNNING` status.
2. **Copy the SSH info** Vast shows you (host + port) and update `~/.vastrc`.
3. **Bootstrap the box** (one command):
   ```bash
   ./scripts/vast.sh setup
   ```
   This: installs apt/pip deps, clones the repo to `/workspace/FMODetect-v2`, downloads the H5 + checkpoint via `gdown`. ~10–15 min.

4. **(Optional) push any local edits** that aren't on GitHub yet:
   ```bash
   ./scripts/vast.sh push
   ```

5. **Train**:
   ```bash
   ./scripts/vast.sh train
   ```
   Output streams live in your terminal. Default: resume from epoch 29, train to epoch 45 with `configs/vast.yaml` (bs=64).

6. **Pull experiments back** (during or after training):
   ```bash
   ./scripts/vast.sh pull
   ```

7. **Quick remote inference on a local image**:
   ```bash
   ./scripts/vast.sh infer FMODetect-master/example/ex1_im.png FMODetect-master/example/ex1_bgr.png
   # → experiments/infer_remote/ex1_im_out.png
   ```

8. **DESTROY the instance** when done (Vast web UI). Keep the meter off.

### Common operations

```bash
./scripts/vast.sh ssh                      # open shell on the box
./scripts/vast.sh run "nvidia-smi"         # one-shot command
./scripts/vast.sh tail                     # which run is latest? what checkpoints?
./scripts/vast.sh train --epochs 60        # different target epoch count
```

---

## Workflow B — VS Code Remote-SSH

If you live in VS Code:

1. Install the **Remote-SSH** extension.
2. `Cmd/Ctrl-Shift-P` → "Remote-SSH: Add New SSH Host…" → paste the Vast ssh command.
3. Connect. The status bar shows `SSH: root@1.2.3.4`.
4. `Open Folder…` → `/workspace/FMODetect-v2` (clone it first with `git clone https://github.com/jai-krishna-0921/FMODetect-v2.git /workspace/FMODetect-v2`).
5. Edit files normally; they save directly on the remote. Open a terminal in VS Code (it's already on the remote) and `python scripts/train.py --config configs/vast.yaml --resume ...`.

You can also use Workflow A's `vast.sh setup` once from your local terminal first to do the gdown/install boilerplate, then switch to VS Code Remote-SSH for iterative work.

---

## Budget (RTX PRO 6000 @ $1.20/h)

| Step | Time | Cost |
|---|---|---|
| `vast.sh setup` (apt + pip + clone + gdown 16 GB H5) | ~10–15 min | $0.20–0.30 |
| `vast.sh train` (16 epochs from epoch 29 to 45) | ~25 min | $0.50 |
| Smoke inference + pull artifacts | ~3 min | $0.06 |
| **Total** | **~45 min** | **~$0.85** |

If `vast.sh setup` already ran in a previous session and you still have the same instance, it skips the gdown step (files are cached on `/workspace/data/`).

If you destroy the instance, the cached H5 is gone; budget the full gdown time again on the next box.
