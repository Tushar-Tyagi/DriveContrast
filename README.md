# AutoVLA 4D

This repository contains the refactored framework to implement a 4D (spatiotemporal) visual backbone for the AutoVLA model.

Instead of a static image encoder (e.g., DinoV2 or SigLIP), this repository implements `MCG-NJU/videomae-base` to ingest 16-frame Video Action features, seamlessly integrating with the open-source `Qwen/Qwen2.5-VL-3B-Instruct` Vision-Language Model.

## Getting Started

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Generate the K-disk Action Vocabulary prior to training.
```bash
python data/tokenizer.py --dataset_dir /path/to/impromptu
```

3. Train the model using Supervised Fine Tuning (SFT). The provided training script balances cross-entropy action-prediction loss over a predefined horizon using QLoRA optimization:
```bash
python scripts/train.py --batch_size 4 --lr 1e-5
```

4. Evaluate performance via PDMS tracking in `navsim`:
```bash
python scripts/eval.py
```

## Dataset Setup: Waymo Subset (< 250 GB)

The full Waymo Open Dataset alongside the full Impromptu-VLA QA dataset is massive. To accommodate constrained storage limits (e.g., under 250 GB), this project utilizes a subset of the pre-processed trajectory data provided by [Impromptu-VLA](https://github.com/ahydchh/Impromptu-VLA).

To download only a fraction of the data directly from HuggingFace, ensuring you stay within hardware limits:

1. Create a Python script (e.g., `download_waymo_subset.py`) to selectively download a few `.tar` shards:

```python
from huggingface_hub import hf_hub_download
import os

repo_id = "aaaaaap/unstructed"
save_dir = "./data/waymo_subset"
os.makedirs(save_dir, exist_ok=True)

# Select 2 training shards and 1 validation shard. 
# This provides a mix of examples while keeping the total footprint under ~50-100GB
shards_to_download = [
    "waymo/waymo_train_shard_0000.tar",
    "waymo/waymo_train_shard_0001.tar",
    "waymo/waymo_val_shard_0000.tar"
]

for shard in shards_to_download:
    print(f"Downloading {shard}...")
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=shard,
        local_dir=save_dir
    )
print("Finished downloading Waymo subset.")
```

2. Run the script:
```bash
python download_waymo_subset.py
```

3. Then point the `--dataset_dir` in the tokenizer/training scripts to `./data/waymo_subset`.
