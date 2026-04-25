import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from data.dataset import ImpromptuVLADataset
from data.tokenizer import ActionTokenizer
from models.vla import AutoVLA4D
from models.baseline_vla import BaselineVLA
from peft import PeftModel

def visualize_comparison(image, gt_actions, baseline_actions, autovla_actions, output_path):
    """
    Left: Front-camera image
    Right: Bird-Eye-View (top-down) trajectory plot comparing GT, Baseline, and AutoVLA4D
    """
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    else:
        img_arr = image

    # Camera View
    ax_cam.imshow(img_arr)
    ax_cam.set_title("Front Camera", fontsize=12, fontweight='bold')
    ax_cam.axis('off')

    # BEV View
    # X = forward (up), Y = lateral (positive = left)
    # We plot lateral on x-axis and forward on y-axis
    def plot_traj(ax, actions, color, label, marker, edge_color, zorder):
        fwd = actions[:, 0]
        lat = -actions[:, 1] # negate for intuitive plot (positive left)
        ax.plot(lat, fwd, color=color, linestyle='-', linewidth=2, alpha=0.6)
        ax.scatter(lat, fwd, c=color, s=50, label=label, alpha=0.9, edgecolors=edge_color, linewidths=0.8, zorder=zorder)

    plot_traj(ax_bev, gt_actions, 'red', 'Ground Truth', 'o', 'darkred', 5)
    plot_traj(ax_bev, baseline_actions, 'dodgerblue', 'Baseline', 's', 'navy', 6)
    plot_traj(ax_bev, autovla_actions, 'orange', 'AutoVLA4D', 'D', 'darkorange', 7)

    # Ego vehicle
    ax_bev.scatter([0], [0], c='green', s=120, marker='^', label='Ego Vehicle', edgecolors='darkgreen', linewidths=1, zorder=8)

    ax_bev.set_xlabel("Lateral offset (m)", fontsize=10)
    ax_bev.set_ylabel("Forward distance (m)", fontsize=10)
    ax_bev.set_title("Bird-Eye-View Trajectory Comparison", fontsize=12, fontweight='bold')
    ax_bev.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

def load_autovla4d(device):
    print("Loading AutoVLA4D...")
    model = AutoVLA4D(use_vanilla_backbone=False)
    
    base_path = "models/autoVLA4d-MAE"
    
    # Load LoRA
    lora_path = f"{base_path}/lora_adapter"
    adapter_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(lora_path, "adapter_model.bin")
    
    has_lora_weights = False
    if os.path.exists(adapter_safetensors) and os.path.getsize(adapter_safetensors) > 1024:
        has_lora_weights = True
    elif os.path.exists(adapter_bin) and os.path.getsize(adapter_bin) > 1024:
        has_lora_weights = True

    if os.path.exists(lora_path) and has_lora_weights:
        try:
            model.vlm = PeftModel.from_pretrained(model.vlm, os.path.abspath(lora_path), is_trainable=False, local_files_only=True)
        except Exception as e:
            print(f"Warning: Failed to load adapter weights from {lora_path}. Error: {e}. Running base Qwen model.")
    else:
        print(f"Warning: LFS pointer or missing weights in {lora_path}. Skipping LoRA load.")
    
    # Load Projector
    proj_path = f"{base_path}/projector_weights.pt"
    if os.path.exists(proj_path):
        model.projector.load_state_dict(torch.load(proj_path, map_location=device))
        
    # Load Action Head
    head_path = f"{base_path}/action_head_weights.pt"
    if os.path.exists(head_path):
        model.action_head.load_state_dict(torch.load(head_path, map_location=device))
        
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load("data/action_centers.pt")

    # Datasets
    # We want a few images from train and a few from test/val
    train_dataset = ImpromptuVLADataset(data_dir="data", subset="Unconventional Dynamic Obstacles", split="train")
    val_dataset = ImpromptuVLADataset(data_dir="data", subset="Unconventional Dynamic Obstacles", split="val")

    # Select 2 random indices from each
    np.random.seed(42)
    train_indices = np.random.choice(len(train_dataset), size=2, replace=False)
    val_indices = np.random.choice(len(val_dataset), size=2, replace=False)

    samples = []
    for idx in train_indices:
        samples.append(("train", idx, train_dataset[idx]))
    for idx in val_indices:
        samples.append(("val", idx, val_dataset[idx]))

    # Image stats for baseline
    image_mean = torch.tensor(train_dataset.processor.image_mean).view(1, 3, 1, 1)
    image_std = torch.tensor(train_dataset.processor.image_std).view(1, 3, 1, 1)

    # Dictionary to hold predictions
    predictions = {
        "baseline": [],
        "autovla4d": []
    }

    # 1. Run Baseline
    print("\n--- Running Baseline ---")
    baseline_model = BaselineVLA.from_pretrained("models/baseline/final")
    baseline_model.action_head.to(device)
    baseline_model.eval()

    with torch.no_grad():
        for split, idx, sample in samples:
            batch = {
                "pixel_values": sample["pixel_values"].unsqueeze(0),
                "continuous_actions": sample["continuous_actions"].unsqueeze(0)
            }
            target_tokens = tokenizer.encode(batch["continuous_actions"])
            horizon = target_tokens.size(1)

            qwen_inputs = baseline_model.build_inputs(batch, device, image_mean, image_std, horizon=horizon)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                action_logits = baseline_model(horizon=horizon, **qwen_inputs)
            
            predicted_tokens = torch.argmax(action_logits, dim=-1)
            predicted_actions = tokenizer.decode(predicted_tokens.cpu()).numpy()[0]
            predictions["baseline"].append(predicted_actions)
    
    del baseline_model
    torch.cuda.empty_cache()

    # 2. Run AutoVLA4D
    print("\n--- Running AutoVLA4D ---")
    autovla_model = load_autovla4d(device)
    
    with torch.no_grad():
        for i, (split, idx, sample) in enumerate(samples):
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            continuous_actions = sample["continuous_actions"].unsqueeze(0).to(device)
            target_tokens = tokenizer.encode(continuous_actions)
            horizon = target_tokens.size(1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = autovla_model(pixel_values=pixel_values)
                action_logits = logits[:, -horizon:, :]
            
            predicted_tokens = torch.argmax(action_logits, dim=-1)
            predicted_actions = tokenizer.decode(predicted_tokens.cpu()).numpy()[0]
            predictions["autovla4d"].append(predicted_actions)

    del autovla_model
    torch.cuda.empty_cache()

    # 3. Visualize
    print("\n--- Generating Visualizations ---")
    for i, (split, idx, sample) in enumerate(samples):
        # We need the raw front camera image
        video_path = sample["video_path"]
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to white image if video cant be read
            frame_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 255
            
        gt_actions = sample["continuous_actions"].numpy()
        baseline_actions = predictions["baseline"][i]
        autovla_actions = predictions["autovla4d"][i]
        
        out_path = f"visualizations/comparison_{split}_{idx}.png"
        visualize_comparison(frame_rgb, gt_actions, baseline_actions, autovla_actions, out_path)

if __name__ == "__main__":
    main()
