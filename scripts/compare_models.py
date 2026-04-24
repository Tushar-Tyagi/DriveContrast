"""
Baseline (Qwen2.5-VL) robustness comparison across 4 augmentation conditions.
Evaluates per-clip metrics and identifies performance under perturbations.

Usage:
    python scripts/compare_models.py --dataset_dir data --split val
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.augmentations import apply_augmentation

import cv2
from PIL import Image
from transformers import AutoImageProcessor
import glob


class SimpleVideoDataset:
    """
    Lightweight cv2-based dataset (no decord dependency).
    Loads 16-frame clips from mp4 + paired .npy actions.
    """
    def __init__(self, data_dir, subset="Unconventional Dynamic Obstacles",
                 num_frames=16, resolution=224, split="val",
                 processor_name="MCG-NJU/videomae-base"):
        self.num_frames = num_frames
        self.resolution = resolution
        self.video_files = sorted(glob.glob(
            os.path.join(data_dir, subset, split, "*.mp4")
        ))
        self.processor = AutoImageProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).resize(
                (self.resolution, self.resolution), Image.BILINEAR
            )
            frames.append(pil_img)
        cap.release()

        # Sample / pad to num_frames
        total = len(frames)
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        # Process through VideoMAE processor
        processed = self.processor(frames, return_tensors="pt")["pixel_values"]
        if processed.dim() == 5:
            processed = processed.squeeze(0)  # (16, 3, 224, 224)

        # Load actions
        base = os.path.splitext(video_path)[0]
        npy_path = base + ".npy"
        if os.path.exists(npy_path):
            actions = torch.tensor(np.load(npy_path), dtype=torch.float32)
        else:
            actions = torch.zeros((10, 2), dtype=torch.float32)

        return {
            "pixel_values": processed,
            "continuous_actions": actions,
        }


class ActionTokenizer:
    """Minimal tokenizer that loads pre-fitted K-means centers."""
    def __init__(self, vocab_size=2048):
        self.vocab_size = vocab_size
        self.centers = None

    def load(self, path):
        self.centers = torch.load(path, weights_only=True)
        print(f"Loaded {self.centers.shape[0]} action centers from {path}")

    def encode(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        original_shape = continuous_actions.shape[:-1]
        flat = continuous_actions.view(-1, continuous_actions.shape[-1])
        # If centers have different dim than actions, truncate to match
        centers = self.centers.to(continuous_actions.device)
        action_dim = flat.shape[-1]
        center_dim = centers.shape[-1]
        if action_dim != center_dim:
            dim = min(action_dim, center_dim)
            flat = flat[:, :dim]
            centers = centers[:, :dim]
        distances = torch.cdist(flat.float(), centers.float())
        tokens = torch.argmin(distances, dim=-1)
        return tokens.view(original_shape)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.centers[tokens]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_pdms(nc, dac, ep, ttc, c):
    return (nc * dac) * ((5 * ttc + 5 * ep + 2 * c) / 12)


def score_comfort(traj):
    """Jerk-based comfort proxy. traj: (H, D)"""
    if len(traj) < 3:
        return 1.0
    vel = np.diff(traj[:, :2], axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.linalg.norm(acc, axis=1).mean()
    return float(np.clip(1.0 - jerk / 10.0, 0.0, 1.0))


def score_ep(pred_traj, gt_traj):
    """Ego progress: closeness of final predicted position to GT endpoint."""
    final_pred = pred_traj[-1, :2]
    final_gt = gt_traj[-1, :2]
    init_gt = gt_traj[0, :2]

    max_progress = np.linalg.norm(final_gt - init_gt) + 1e-6
    actual = max_progress - np.linalg.norm(final_pred - final_gt)
    return float(np.clip(actual / max_progress, 0.0, 1.0))


def l2_trajectory_error(pred_traj, gt_traj):
    """Mean L2 distance between predicted and GT waypoints."""
    return float(np.mean(np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=1)))


# ── Baseline Model ───────────────────────────────────────────────────────────

def load_baseline_model(device):
    """Load Qwen2.5-VL-3B-Instruct as the baseline."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("Loading Qwen2.5-VL-3B-Instruct baseline...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model.eval()
    return model, processor


def build_qwen_inputs(pixel_values_batch, processor, device):
    """
    Convert VideoMAE-preprocessed frames to Qwen VL inputs.
    Takes the middle frame of the 16-frame clip as the single image input.
    """
    VIDEOMAE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    VIDEOMAE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    B = pixel_values_batch.shape[0]

    # Pick middle frame: (B, 16, 3, 224, 224) → (B, 3, 224, 224)
    mid = pixel_values_batch.shape[1] // 2
    frames = pixel_values_batch[:, mid, :, :, :].cpu()

    # Undo VideoMAE normalization → [0, 1]
    frames = frames * VIDEOMAE_STD + VIDEOMAE_MEAN
    frames = frames.clamp(0.0, 1.0)

    # (B, 3, H, W) → (B, H, W, 3) uint8
    images_np = (
        frames.permute(0, 2, 3, 1)
        .mul(255)
        .byte()
        .numpy()
    )

    conversations = []
    for _ in range(B):
        conversations.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Predict the future trajectory of the ego vehicle."},
            ],
        }])

    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]

    inputs = processor(
        text=texts,
        images=[images_np[i] for i in range(B)],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


# ── Evaluation Loop ──────────────────────────────────────────────────────────

def evaluate_baseline_per_clip(model, dataset, tokenizer, processor, device, augmentation="unaltered"):
    """
    Run baseline evaluation on every clip with a specific augmentation.
    Returns a list of per-clip result dicts.
    """
    model.eval()
    device_type = device.type
    criterion = nn.CrossEntropyLoss(reduction="none")
    results = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Baseline [{augmentation}]")):
            # Get clip ID from the dataset
            video_path = dataset.video_files[idx]
            clip_id = os.path.splitext(os.path.basename(video_path))[0]

            pixel_values = batch["pixel_values"]  # (1, 16, 3, 224, 224)
            continuous_actions = batch["continuous_actions"].to(device)  # (1, H, D)

            # Apply augmentation
            pixel_values = apply_augmentation(pixel_values, augmentation)
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

            # Encode GT actions
            target_tokens = tokenizer.encode(continuous_actions)  # (1, H)
            horizon = target_tokens.size(1)

            # Build Qwen inputs from (possibly augmented) frames
            qwen_inputs = build_qwen_inputs(pixel_values, processor, device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(
                    **qwen_inputs,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
                action_logits_full = model.lm_head(hidden_states)  # (1, seq_len, vocab)
                action_logits = action_logits_full[:, -horizon:, :2048]  # (1, H, 2048)

                loss_per_step = criterion(
                    action_logits.reshape(-1, 2048),
                    target_tokens.view(-1),
                )
                loss = loss_per_step.mean().item()

            # Decode predictions
            predicted_tokens = torch.argmax(action_logits, dim=-1)  # (1, H)
            predicted_actions = tokenizer.decode(predicted_tokens.cpu()).numpy()  # (1, H, D)
            gt_actions = continuous_actions.cpu().numpy()  # (1, H, D)

            pred = predicted_actions[0]  # (H, D)
            gt = gt_actions[0]  # (H, D)

            # Score
            nc = 1.0
            dac = 1.0
            ep = score_ep(pred, gt)
            ttc = 1.0
            c = score_comfort(pred)
            pdms = compute_pdms(nc, dac, ep, ttc, c)
            l2_err = l2_trajectory_error(pred, gt)

            results.append({
                "clip_id": clip_id,
                "condition": augmentation,
                "EP": round(ep, 4),
                "C": round(c, 4),
                "PDMS": round(pdms, 4),
                "loss": round(loss, 4),
                "l2_error": round(l2_err, 4),
                "pred_actions": pred.tolist(),
                "gt_actions": gt.tolist(),
            })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

CONDITIONS = ["unaltered", "noise_injection", "cutout", "frame_drop"]


def print_summary(all_results):
    """Print a summary table grouped by condition."""
    print("\n" + "=" * 70)
    print(f"{'Condition':<20} {'EP':>8} {'Comfort':>8} {'PDMS':>8} {'L2 Err':>8} {'Loss':>8}")
    print("-" * 70)

    for cond in CONDITIONS:
        cond_results = [r for r in all_results if r["condition"] == cond]
        if not cond_results:
            continue
        avg_ep = np.mean([r["EP"] for r in cond_results])
        avg_c = np.mean([r["C"] for r in cond_results])
        avg_pdms = np.mean([r["PDMS"] for r in cond_results])
        avg_l2 = np.mean([r["l2_error"] for r in cond_results])
        avg_loss = np.mean([r["loss"] for r in cond_results])
        print(f"{cond:<20} {avg_ep:>8.4f} {avg_c:>8.4f} {avg_pdms:>8.4f} {avg_l2:>8.2f} {avg_loss:>8.4f}")

    print("=" * 70)

    # Per-condition degradation relative to unaltered
    unaltered_results = [r for r in all_results if r["condition"] == "unaltered"]
    if unaltered_results:
        baseline_pdms = np.mean([r["PDMS"] for r in unaltered_results])
        print(f"\nPDMS degradation relative to unaltered (baseline PDMS={baseline_pdms:.4f}):")
        for cond in CONDITIONS[1:]:
            cond_results = [r for r in all_results if r["condition"] == cond]
            if cond_results:
                cond_pdms = np.mean([r["PDMS"] for r in cond_results])
                delta = cond_pdms - baseline_pdms
                pct = (delta / baseline_pdms) * 100 if baseline_pdms > 0 else 0
                print(f"  {cond:<20} ΔPDMS={delta:+.4f}  ({pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Baseline robustness comparison")
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--subset", type=str, default="Unconventional Dynamic Obstacles")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--action_centers", type=str, default="data/action_centers.pt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Limit number of clips (for quick testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load(args.action_centers)

    # Load dataset
    dataset = SimpleVideoDataset(
        data_dir=args.dataset_dir,
        subset=args.subset,
        split=args.split,
    )
    if args.max_clips and args.max_clips < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(args.max_clips)))
        # Preserve video_files for clip ID lookup
        dataset.video_files = dataset.dataset.video_files[:args.max_clips]
    print(f"Evaluation set: {len(dataset)} clips")

    # Load baseline
    model, processor = load_baseline_model(device)

    # Run evaluation across all conditions
    all_results = []
    for condition in CONDITIONS:
        print(f"\n{'─' * 50}")
        print(f"Evaluating condition: {condition}")
        print(f"{'─' * 50}")

        results = evaluate_baseline_per_clip(
            model, dataset, tokenizer, processor, device,
            augmentation=condition,
        )
        all_results.extend(results)

    # Save raw results (without pred/gt arrays for readability)
    results_slim = []
    for r in all_results:
        slim = {k: v for k, v in r.items() if k not in ("pred_actions", "gt_actions")}
        results_slim.append(slim)

    output_path = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results_slim, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save full results (with trajectories) for visualization
    full_output_path = os.path.join(args.output_dir, "baseline_comparison_full.json")
    with open(full_output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print_summary(all_results)

    # Save summary to text file
    summary_path = os.path.join(args.output_dir, "baseline_comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Baseline Robustness Comparison — {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {args.subset}/{args.split} ({len(dataset)} clips)\n\n")

        f.write(f"{'Condition':<20} {'EP':>8} {'Comfort':>8} {'PDMS':>8} {'L2 Err':>8} {'Loss':>8}\n")
        f.write("-" * 70 + "\n")
        for cond in CONDITIONS:
            cond_results = [r for r in all_results if r["condition"] == cond]
            if cond_results:
                f.write(f"{cond:<20} "
                        f"{np.mean([r['EP'] for r in cond_results]):>8.4f} "
                        f"{np.mean([r['C'] for r in cond_results]):>8.4f} "
                        f"{np.mean([r['PDMS'] for r in cond_results]):>8.4f} "
                        f"{np.mean([r['l2_error'] for r in cond_results]):>8.2f} "
                        f"{np.mean([r['loss'] for r in cond_results]):>8.4f}\n")

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
