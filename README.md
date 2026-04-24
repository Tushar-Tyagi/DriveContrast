# AutoVLA 4D

This repository contains the refactored framework to implement a 4D (spatiotemporal) visual backbone for the AutoVLA model.

Instead of a static image encoder (e.g., DinoV2 or SigLIP), this repository implements `MCG-NJU/videomae-base` to ingest 16-frame Video Action features, seamlessly integrating with the open-source `Qwen/Qwen2.5-VL-3B-Instruct` Vision-Language Model.

## Models

### Baseline: Qwen2.5-VL-3B-Instruct (Vanilla)

The baseline uses `Qwen/Qwen2.5-VL-3B-Instruct` with its **native static vision tower** — a single-frame image encoder that processes one frame at a time. During evaluation, the middle frame of a 16-frame clip is extracted, un-normalized from VideoMAE space, and fed to the model as a standard image prompt alongside the text instruction:

> *"Predict the future trajectory of the ego vehicle."*

The model's output is decoded into a trajectory as follows:

1. Extract the **last hidden states** from the decoder's final layer
2. Project through the **LM head** (hidden dim → full Qwen vocab of ~151k tokens)
3. **Slice** to only the last `H=10` timesteps and the first 2048 logits (matching the K-means action codebook size)
4. **Argmax** over the 2048 action tokens per timestep
5. **Decode** token IDs back to continuous `[forward_m, lateral_m]` BEV coordinates via the pre-fitted K-means cluster centers

Since the model was never trained to map its vocabulary to driving actions, it is effectively a zero-shot baseline — the first 2048 token positions in Qwen's vocabulary are repurposed as an action codebook without any fine-tuning.

- **Input**: Single image (middle frame of the clip) + text prompt
- **Vision encoder**: Qwen's built-in ViT (static, 2D)
- **No fine-tuning**: Used off-the-shelf with zero-shot prompting
- **Evaluation script**: `scripts/eval_baseline.py`

### Ours: AutoVLA 4D

AutoVLA 4D replaces the static vision backbone with `MCG-NJU/videomae-base`, a spatiotemporal transformer that processes all **16 frames** of each clip. VideoMAE features are mapped to Qwen's hidden dimension via a learned 2-layer MLP projector, then decoded through the Qwen language model (fine-tuned with QLoRA) and a discrete action head.

- **Input**: 16-frame video clip (2 FPS × 8 seconds)
- **Vision encoder**: VideoMAE-base (4D spatiotemporal)
- **Fine-tuned**: QLoRA on Qwen + trained projector + action head
- **Evaluation script**: `scripts/eval.py`

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

7. **Visualize Results**:
   Generate dual-panel visualizations (raw camera view + top-down BEV trajectory plot) for the evaluated clips:
   ```bash
   python visualize_example.py
   ```
   Or use the `visualization_suite.py` programmatically to compare Ground Truth (red) against Baseline Predictions (blue).

## Dataset Setup: Waymo Subset (< 300 GB)

The full Waymo Open Dataset and full [Impromptu-VLA](https://github.com/ahydchh/Impromptu-VLA) QA data are large. This project uses a small subset from HuggingFace ([aaaaaap/unstructed](https://huggingface.co/datasets/aaaaaap/unstructed)) to stay under ~50–100 GB.

1. **Download** (step 2 above):
   ```bash
   python download_waymo_subset.py
   ```
   Creates `data/waymo_subset/` with `waymo/waymo_train_shard_*.tar` and `waymo/waymo_val_shard_*.tar`.

2. **Extract** (step 3 above) so the dataloader sees `.mp4` and `.npy` per sample. Then use the same `--dataset_dir data/waymo_subset` for the tokenizer and for training.
