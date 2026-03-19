"""
Evaluates the finetuned Qwen2.5-VL baseline (BaselineVLA) on the Waymo val split

Usage:
    PYTHONPATH=$PWD python3 scripts/eval_baseline_finetuned.py \
        --checkpoint_dir models/baseline/final \
        --dataset_dir data/waymo_subset \
        --action_centers data/action_centers.pt \
        --split val
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline_vla import BaselineVLA
from data.dataset import ImpromptuVLADataset
from data.tokenizer import ActionTokenizer


def compute_pdms(nc, dac, ep, ttc, c):
    return (nc * dac) * ((5 * ttc + 5 * ep + 2 * c) / 12)


def score_ep(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """Ego-progress: proximity of final predicted waypoint to GT endpoint."""
    final_pred = pred_traj[-1, :2]
    final_gt   = gt_traj[-1,  :2]
    init_gt    = gt_traj[0,   :2]
    max_prog   = np.linalg.norm(final_gt - init_gt) + 1e-6
    actual     = max_prog - np.linalg.norm(final_pred - final_gt)
    return float(np.clip(actual / max_prog, 0.0, 1.0))


def score_comfort(traj: np.ndarray) -> float:
    """Jerk-based comfort. Requires >= 4 waypoints."""
    if len(traj) < 4:
        return 1.0
    vel  = np.diff(traj[:, :2], axis=0)
    acc  = np.diff(vel,         axis=0)
    jerk = np.diff(acc,         axis=0)
    J    = np.linalg.norm(jerk, axis=1).mean()
    return float(np.clip(1.0 - J / 10.0, 0.0, 1.0))


def evaluate(model, dataloader, tokenizer, criterion, device, image_mean, image_std, print_trajectories):
    model.eval()
    total_loss = 0.0
    metrics    = {"NC": [], "DAC": [], "EP": [], "TTC": [], "C": [], "PDMS": []}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            continuous_actions = batch["continuous_actions"].to(device)
            target_tokens      = tokenizer.encode(continuous_actions)      # (B, H)
            horizon            = target_tokens.size(1)

            qwen_inputs = model.build_inputs(
                batch, device, image_mean, image_std, horizon=horizon
            )

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                action_logits = model(horizon=horizon, **qwen_inputs)      # (B, H, 2048)
                loss = criterion(
                    action_logits.reshape(-1, 2048),
                    target_tokens.view(-1).long(),
                )

            total_loss += loss.item()
            n_printed = 0
            predicted_tokens  = torch.argmax(action_logits, dim=-1)        # (B, H)
            predicted_actions = tokenizer.decode(predicted_tokens.cpu()).numpy()
            gt_actions        = continuous_actions.cpu().numpy()

            for b in range(predicted_actions.shape[0]):
                pred = predicted_actions[b]
                gt   = gt_actions[b]
                ep   = score_ep(pred, gt)
                c    = score_comfort(pred)
                pdms = compute_pdms(nc=1.0, dac=1.0, ep=ep, ttc=1.0, c=c)

                metrics["NC"].append(1.0)
                metrics["DAC"].append(1.0)
                metrics["EP"].append(ep)
                metrics["TTC"].append(1.0)
                metrics["C"].append(c)
                metrics["PDMS"].append(pdms)

                if print_trajectories and n_printed < 3:
                    print(f"\n── Sample {n_printed + 1} ──────────────────────────")
                    print(f"  GT   waypoints: {np.round(gt[:, :2], 3).tolist()}")
                    print(f"  Pred waypoints: {np.round(pred[:, :2], 3).tolist()}")
                    print(f"  EP={ep:.3f}  C={c:.3f}  PDMS={pdms:.3f}")
                    n_printed += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                PDMS=f"{np.mean(metrics['PDMS']):.3f}",
            )

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["loss"] = total_loss / len(dataloader)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",    type=str, default="data/waymo_subset")
    parser.add_argument("--subset",         type=str, default="Unconventional Dynamic Obstacles")
    parser.add_argument("--split",          type=str, default="val")
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--num_workers",    type=int, default=2)
    parser.add_argument("--action_centers", type=str, default="data/action_centers.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="models/baseline/final")
    parser.add_argument("--print_trajectories", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load(args.action_centers)
    dataset = ImpromptuVLADataset(
        data_dir=args.dataset_dir, subset=args.subset, split=args.split
    )
    image_mean = torch.tensor(dataset.processor.image_mean).view(1, 3, 1, 1)
    image_std  = torch.tensor(dataset.processor.image_std).view(1, 3, 1, 1)
    print(f"VideoMAE norm  mean={dataset.processor.image_mean}  std={dataset.processor.image_std}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
    )
    print(f"Eval samples: {len(dataset)}")
    print(f"Loading checkpoint from {args.checkpoint_dir} ...")
    model = BaselineVLA.from_pretrained(args.checkpoint_dir)
    model.action_head.to(device)
    criterion = nn.CrossEntropyLoss()

    print("\nRunning evaluation ...")
    results = evaluate(model, dataloader, tokenizer, criterion, device, image_mean, image_std, print_trajectories=args.print_trajectories)

    print("\n── Results ────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<6}: {v:.4f}")

    out = f"models/eval_results_baseline_finetuned_{args.split}.txt"
    os.makedirs("models", exist_ok=True)
    with open(out, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()