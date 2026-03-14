import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import Dinov2Model # Dummy required for older HF environments

from models.vla import AutoVLA4D
from data.dataset import ImpromptuVLADataset
from data.tokenizer import ActionTokenizer

def configure_qlora(model):
    """
    Configures QLoRA on the language model backbone (Qwen2.5) to isolate trainable parameters and save VRAM.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA on the underlying Qwen causal model logic
    model.vlm = get_peft_model(model.vlm, lora_config)
    
    # Ensure our custom components remain trainable
    model.projector.requires_grad_(True)
    model.action_head.requires_grad_(True)
    
    # We freeze VideoMAE encoder to preserve spatiotemporal priors
    model.vision_encoder.requires_grad_(False)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Action Tokenizer
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.load("data/action_centers.pt") # Must be generated prior

    # 2. Init Dataset
    dataset = ImpromptuVLADataset(data_dir="data/impromptu", split="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    # 3. Model init
    model = AutoVLA4D(use_vanilla_backbone=False)
    model = configure_qlora(model)
    model.to(device)

    # 4. Training configuration 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-5, 
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    EPOCHS = 3
    
    print("\nStarting Training (SFT)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # (B, 16, 3, 224, 224)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            
            # (B, Horizon, Action_Dim) -> e.g. (B, 10, 3) 
            continuous_actions = batch["continuous_actions"].to(device)
            
            # Discretize ground truth actions into (B, Horizon) token indices
            with torch.no_grad():
                target_tokens = tokenizer.encode(continuous_actions)
            
            # Forward pass: BFloat16 precision for efficiency
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                # We expect the model's output sequence length to ideally match the action horizon
                # Though in reality VVLMs output based on patch tokens. We assume the output logits
                # are aligned to directly map or that we pool patch features to sequence length.
                
                # Output shape: (B, seq_length, vocab_size)
                logits = model(pixel_values=pixel_values) 
                
                # Extract the last H tokens representing action predictions
                horizon = target_tokens.size(1)
                action_logits = logits[:, -horizon:, :] # (B, H, Vocab_Size)
                
                # Reshape for CE Loss -> (B * H, Vocab) & Target -> (B * H)
                loss = criterion(action_logits.reshape(-1, 2048), target_tokens.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    # Save components
    print("Saving Projector and Action Head weights...")
    torch.save(model.projector.state_dict(), "models/projector_weights.pt")
    torch.save(model.action_head.state_dict(), "models/action_head_weights.pt")
    print("Saving completed.")

if __name__ == "__main__":
    main()
