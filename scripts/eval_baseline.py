import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from data.dataset import ImpromptuVLADataset
from data.tokenizer import ActionTokenizer


def load_qwen_model(device):
    print("Loading Qwen2.5-VL-3B-Instruct with native vision tower...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model.eval()
    return model, processor


def compute_pdms(nc, dac, ep, ttc, c):
    return (nc * dac) * ((5 * ttc + 5 * ep + 2 * c) / 12)


def score_comfort(traj):
    """Jerk-based comfort proxy. traj: (H, 3)"""
    if len(traj) < 3:
        return 1.0
    vel  = np.diff(traj[:, :2], axis=0)
    acc  = np.diff(vel, axis=0)
    jerk = np.linalg.norm(acc, axis=1).mean()
    return float(np.clip(1.0 - jerk / 10.0, 0.0, 1.0))


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


def build_qwen_inputs(batch, processor, device):
    VIDEOMAE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    VIDEOMAE_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    B = batch["pixel_values"].shape[0]

    # (B, 16, 3, 224, 224) → pick middle frame → (B, 3, 224, 224)
    mid = batch["pixel_values"].shape[1] // 2
    frames = batch["pixel_values"][:, mid, :, :, :].cpu()  # <-- this line was missing

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


def evaluate(model, dataloader, tokenizer, criterion, processor, device):
    model.eval()
    device_type = device.type

    total_loss = 0.0
    metrics = {"NC": [], "DAC": [], "EP": [], "TTC": [], "C": [], "PDMS": []}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            continuous_actions = batch["continuous_actions"].to(device)   # (B, H, 3)
            target_tokens      = tokenizer.encode(continuous_actions)     # (B, H)
            horizon            = target_tokens.size(1)

            # Build Qwen-native inputs from raw pixel tensors
            qwen_inputs = build_qwen_inputs(batch, processor, device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(
                    **qwen_inputs,
                    output_hidden_states=True,
                )
                # Use the last hidden state of the final decoder layer
                # shape: (B, seq_len, hidden_dim)
                hidden_states = outputs.hidden_states[-1]

                # Project hidden states → action vocab via the LM head
                # lm_head: hidden_dim → vocab_size (151k for Qwen2.5)
                # We slice action tokens from the tail of the sequence
                action_logits_full = model.lm_head(hidden_states)        # (B, seq_len, vocab)
                action_logits      = action_logits_full[:, -horizon:, :2048]  # (B, H, 2048)

                loss = criterion(
                    action_logits.reshape(-1, 2048),
                    target_tokens.view(-1),
                )

            total_loss += loss.item()

            predicted_tokens  = torch.argmax(action_logits, dim=-1)      # (B, H)
            predicted_actions = tokenizer.decode(predicted_tokens.cpu())  # (B, H, 3)
            gt_actions        = continuous_actions.cpu().numpy()

            B = predicted_actions.shape[0]
            for b in range(B):
                pred = predicted_actions[b]
                gt   = gt_actions[b]

                nc   = 1.0
                dac  = 1.0
                ep   = score_ep(pred, gt)
                ttc  = 1.0
                c    = score_comfort(pred)
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
    parser.add_argument("--dataset_dir",    type=str, default="data/waymo_subset")
    parser.add_argument("--subset",         type=str, default="Unconventional Dynamic Obstacles")
    parser.add_argument("--split",          type=str, default="val")
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--action_centers", type=str, default="data/action_centers.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load(args.action_centers)

    dataset    = ImpromptuVLADataset(data_dir=args.dataset_dir, subset=args.subset, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"Eval set: {len(dataset)} samples")

    model, processor = load_qwen_model(device)
    criterion        = nn.CrossEntropyLoss()

    print("\nRunning evaluation...")
    results = evaluate(model, dataloader, tokenizer, criterion, processor, device)

    print("\n── Results ──────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<6}: {v:.4f}")

    out = f"models/eval_results_qwen_vanilla_{args.split}.txt"
    with open(out, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()