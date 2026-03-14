# AutoVLA 4D (Impromptu VLA)

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
