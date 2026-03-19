"""
scripts/train_baseline.py

Finetunes the Qwen2.5-VL static-image baseline (BaselineVLA) on the Waymo subset.

Usage:
    python scripts/train_baseline.py \
        --dataset_dir data/waymo_subset \
        --action_centers data/action_centers.pt \
        --epochs 3 --batch_size 1 --lr 1e-5
"""

import os
import argparse
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models.baseline_vla import BaselineVLA
from data.dataset import ImpromptuVLADataset
from data.tokenizer import ActionTokenizer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Action tokenizer
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load(args.action_centers)

    # 2. Dataset 
    dataset = ImpromptuVLADataset(
        data_dir=args.dataset_dir, subset=args.subset, split=args.split
    )
    image_mean = torch.tensor(dataset.processor.image_mean).view(1, 3, 1, 1)
    image_std  = torch.tensor(dataset.processor.image_std).view(1, 3, 1, 1)
    print(f"VideoMAE norm  mean={dataset.processor.image_mean}  std={dataset.processor.image_std}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,   
        pin_memory=(device.type == "cuda"),
    )
    print(f"Train samples: {len(dataset)}")

    # 3. Model
    model = BaselineVLA(vocab_size=2048)
    for param in model.qwen.parameters():
        param.requires_grad = False
    # model.apply_lora(r=4, lora_alpha=8, lora_dropout=0.1)
    model.action_head.to(device)
    # optimizer = torch.optim.AdamW(
    #     model.trainable_parameters(), lr=args.lr, weight_decay=0.05
    # )
    optimizer = torch.optim.AdamW(
        model.action_head.parameters(), lr=args.lr, weight_decay=0.05
    )
    total_steps  = len(dataloader) * args.epochs
    warmup_steps = min(100, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    # 5. Training loop
    print("\nStarting baseline SFT …\n")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            continuous_actions = batch["continuous_actions"].to(device)
            with torch.no_grad():
                target_tokens = tokenizer.encode(continuous_actions)       # (B, H)
            horizon = target_tokens.size(1)

            qwen_inputs = model.build_inputs(batch, device, image_mean, image_std, horizon=horizon  )

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                action_logits = model(horizon=horizon, **qwen_inputs)      # (B, H, 2048)
                loss = criterion(
                    action_logits.reshape(-1, 2048),
                    target_tokens.view(-1).long(),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | avg loss: {avg:.4f}")
        
        model.eval()
        val_dataset = ImpromptuVLADataset(
            data_dir=args.dataset_dir, subset=args.subset, split="val"
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0, drop_last=False)
        print(f"\n  Epoch {epoch + 1} trajectory samples:")
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 3:
                    break
                continuous_actions = batch["continuous_actions"].to(device)
                target_tokens = tokenizer.encode(continuous_actions)
                horizon = target_tokens.size(1)
                qwen_inputs = model.build_inputs(
                    batch, device, image_mean, image_std, horizon=horizon
                )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    action_logits = model(horizon=horizon, **qwen_inputs)
                predicted_tokens  = torch.argmax(action_logits, dim=-1)
                predicted_actions = tokenizer.decode(predicted_tokens.cpu()).numpy()
                gt = continuous_actions.cpu().numpy()[0]
                pred = predicted_actions[0]
                print(f"    Sample {i+1}:")
                print(f"      GT  : {np.round(gt[:, :2], 2).tolist()}")
                print(f"      Pred: {np.round(pred[:, :2], 2).tolist()}")
        model.train()
        # ckpt = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        # model.save(ckpt)

    model.save(os.path.join(args.output_dir, "final"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",    type=str, default="data/waymo_subset")
    parser.add_argument("--subset",         type=str, default="Unconventional Dynamic Obstacles")
    parser.add_argument("--split",          type=str, default="train")
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--epochs",         type=int, default=3)
    parser.add_argument("--num_workers",    type=int, default=2)
    parser.add_argument("--action_centers", type=str, default="data/action_centers.pt")
    parser.add_argument("--output_dir",     type=str, default="models/baseline")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()