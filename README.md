# AutoVLA 4D

This repository contains the refactored framework to implement a 4D (spatiotemporal) visual backbone for the AutoVLA model.

Instead of a static image encoder (e.g., DinoV2 or SigLIP), this repository implements `MCG-NJU/videomae-base` to ingest 16-frame Video Action features, seamlessly integrating with the open-source `Qwen/Qwen2.5-VL-3B-Instruct` Vision-Language Model.

## Getting Started

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Waymo subset** (see [Dataset Setup](#dataset-setup-waymo-subset--300-gb) below).
   ```bash
   python download_waymo_subset.py
   ```
   This writes `data/waymo_subset/` with tar shards.

3. **Extract tar shards** into the layout expected by the training dataset (`dataset_dir/subset/split/*.mp4` and `*.npy`):
   ```bash
   python scripts/extract_waymo_subset.py --dataset_dir data/waymo_subset
   ```
   Output: `data/waymo_subset/Unconventional Dynamic Obstacles/train/` and `.../val/` with `{id}.mp4` and `{id}.npy` per sample.

4. **Generate the K-disk action vocabulary** (required once before training). The tokenizer fits from either the tar files or the extracted `.npy` files:
   ```bash
   python data/tokenizer.py --dataset_dir data/waymo_subset --output_file data/action_centers.pt
   ```

5. **Train** with SFT (cross-entropy over action tokens, QLoRA):
   ```bash
   python scripts/train.py --dataset_dir data/waymo_subset --split train --batch_size 4 --lr 1e-5 --epochs 3
   ```
   Optional: `--subset "Unconventional Dynamic Obstacles"` (default) must match the name used in step 3. Checkpoints: `models/projector_weights.pt`, `models/action_head_weights.pt`.

6. **Evaluate** (e.g. PDMS in `navsim`):
   ```bash
   python scripts/eval.py
   ```

## Dataset Setup: Waymo Subset (< 300 GB)

The full Waymo Open Dataset and full [Impromptu-VLA](https://github.com/ahydchh/Impromptu-VLA) QA data are large. This project uses a small subset from HuggingFace ([aaaaaap/unstructed](https://huggingface.co/datasets/aaaaaap/unstructed)) to stay under ~50–100 GB.

1. **Download** (step 2 above):
   ```bash
   python download_waymo_subset.py
   ```
   Creates `data/waymo_subset/` with `waymo/waymo_train_shard_*.tar` and `waymo/waymo_val_shard_*.tar`.

2. **Extract** (step 3 above) so the dataloader sees `.mp4` and `.npy` per sample. Then use the same `--dataset_dir data/waymo_subset` for the tokenizer and for training.
