import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel

from models.vla import AutoVLA4D
from data.dataset import ImpromptuVLADataset
from data.tokenizer import ActionTokenizer


def load_trained_model(args, device):
    model = AutoVLA4D(use_vanilla_backbone=False)
    print(f"Loading LoRA adapter from: {args.lora_adapter}")
    
    model.vlm = PeftModel.from_pretrained(
        model.vlm,
        args.lora_adapter,
        is_trainable=False,
        local_files_only=True,
    )

    model.projector.load_state_dict(
        torch.load(args.projector_weights, map_location=device)
    )
    model.action_head.load_state_dict(
        torch.load(args.action_head_weights, map_location=device)
    )

    model.to(device)
    model.eval()
    return model


def compute_pdms(nc, dac, ep, ttc, c):
    return (nc * dac) * ((5 * ttc + 5 * ep + 2 * c) / 12)


def score_comfort(traj):
    """Jerk-based comfort proxy. traj: (H, 3)"""
    if len(traj) < 3:
        return 1.0
    vel  = np.diff(traj[:, :2], axis=0)
    acc  = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    J = jerk_mag.mean()

    return float(np.clip(1.0 - J / 10.0, 0.0, 1.0))


def score_ep(pred_traj, gt_traj):
    """
    Ego progress: how close the final predicted position is to the GT endpoint.
    pred_traj, gt_traj: (H, 3)
    """
    final_pred = pred_traj[-1, :2]
    final_gt   = gt_traj[-1, :2]
    init_gt    = gt_traj[0, :2]

    max_progress = np.linalg.norm(final_gt - init_gt) + 1e-6
    actual       = max_progress - np.linalg.norm(final_pred - final_gt)
    return float(np.clip(actual / max_progress, 0.0, 1.0))


def evaluate(model, dataloader, tokenizer, criterion, device):
    model.eval()
    device_type = device.type  # "cuda" or "cpu"

    total_loss = 0.0
    metrics = {"NC": [], "DAC": [], "EP": [], "TTC": [], "C": [], "PDMS": []}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            pixel_values       = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            continuous_actions = batch["continuous_actions"].to(device)   # (B, H, 3)
            target_tokens      = tokenizer.encode(continuous_actions)     # (B, H)
            horizon            = target_tokens.size(1)

            # ── Forward ──────────────────────────────────────────────
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits       = model(pixel_values=pixel_values)           # (B, seq_len, vocab)
                action_logits = logits[:, -horizon:, :]                   # (B, H, vocab)
                loss          = criterion(
                    action_logits.reshape(-1, 2048),
                    target_tokens.view(-1)
                )

            total_loss += loss.item()

            # ── Decode predictions ───────────────────────────────────
            predicted_tokens  = torch.argmax(action_logits, dim=-1)      # (B, H)
            predicted_actions = tokenizer.decode(predicted_tokens.cpu())  # (B, H, 3)
            gt_actions        = continuous_actions.cpu().numpy()          # (B, H, 3)

            # ── Score each sample ────────────────────────────────────
            B = predicted_actions.shape[0]
            for b in range(B):
                pred = predicted_actions[b]   # (H, 3)
                gt   = gt_actions[b]          # (H, 3)

                # TODO: replace stubs with real navsim calls
                nc  = 1.0                     # score_nc(pred, gt)
                dac = 1.0                     # score_dac(pred, gt)
                ep  = score_ep(pred, gt)
                ttc = 1.0                     # score_ttc(pred, gt)
                c   = score_comfort(pred)
                pdms = compute_pdms(nc, dac, ep, ttc, c)

                metrics["NC"].append(nc)
                metrics["DAC"].append(dac)
                metrics["EP"].append(ep)
                metrics["TTC"].append(ttc)
                metrics["C"].append(c)
                metrics["PDMS"].append(pdms)

            pbar.set_postfix(loss=loss.item(), PDMS=f"{np.mean(metrics['PDMS']):.3f}")

    n = len(dataloader)
    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["loss"] = total_loss / n
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",      type=str, default="data/waymo_subset")
    parser.add_argument("--subset",           type=str, default="Unconventional Dynamic Obstacles")
    parser.add_argument("--split",            type=str, default="val")
    parser.add_argument("--batch_size",       type=int, default=1)
    parser.add_argument("--action_centers",   type=str, default="data/action_centers.pt")
    parser.add_argument("--lora_adapter",     type=str, default="models/lora_adapter")
    parser.add_argument("--projector_weights",type=str, default="models/projector_weights.pt")
    parser.add_argument("--action_head_weights", type=str, default="models/action_head_weights.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load(args.action_centers)

    # Dataset — same subset/split args as train.py
    dataset    = ImpromptuVLADataset(data_dir=args.dataset_dir, subset=args.subset, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"Eval set: {len(dataset)} samples")

    # Model
    model     = load_trained_model(args, device)
    criterion = nn.CrossEntropyLoss()

    # Run
    print("\nRunning evaluation...")
    results = evaluate(model, dataloader, tokenizer, criterion, device)

    print("\n── Results ──────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<6}: {v:.4f}")

    # Save
    out = f"models/eval_results_{args.split}.txt"
    with open(out, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()