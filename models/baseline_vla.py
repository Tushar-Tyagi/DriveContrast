"""
models/baseline_vla.py

Wraps Qwen2.5-VL-3B-Instruct + DiscreteActionHead into a single nn.Module.
This is the static-image baseline, contrasted against AutoVLA4D (VideoMAE backbone).
"""

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from models.action_head import DiscreteActionHead

PROMPT = (
    "You are an autonomous driving system operating in ego-centric coordinates. "
    "Predict the next 10 waypoints (x, y) in metres over a 5 second horizon at 2 Hz."
    "Each waypoint represents the ego vehicle's future position relative to its current location."
)


class BaselineVLA(nn.Module):
    """
    Qwen2.5-VL-3B-Instruct with a trainable DiscreteActionHead.

    Forward pass:
        inputs: output of BaselineVLA.build_inputs() — already on device
        returns: action_logits  (B, horizon, vocab_size)
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self, vocab_size: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size

        print(f"Loading {self.MODEL_ID} …")
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

        hidden_dim = self.qwen.config.text_config.hidden_size
        self.action_head = DiscreteActionHead(
            hidden_dim=hidden_dim, vocab_size=vocab_size
        )

    # LoRA 
    def apply_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        """Wraps self.qwen with QLoRA. Call once before training."""
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.qwen = get_peft_model(self.qwen, config)
        self.qwen.print_trainable_parameters()

    # def build_inputs(
    #     self, batch: dict, device: torch.device,
    #     image_mean: torch.Tensor, image_std: torch.Tensor,  horizon: int = 10,
    # ) -> dict:
    #     """
    #     Converts a DataLoader batch into Qwen processor inputs
    #     Args:
    #         batch:      DataLoader batch; "pixel_values" is (B, T, 3, H, W)
    #     Returns:
    #         Dict of tensors on `device`.
    #     """
    #     B   = batch["pixel_values"].shape[0]
    #     mid = batch["pixel_values"].shape[1] // 2

    #     frames = batch["pixel_values"][:, mid].cpu()                    # (B,3,H,W)
    #     frames = (frames * image_std + image_mean).clamp(0.0, 1.0)     # undo norm
    #     images_np = frames.permute(0, 2, 3, 1).mul(255).byte().numpy() # (B,H,W,3)
    #     action_slots = " " + " ".join(["."] * horizon)
    #     conversations = [
    #         [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": PROMPT},
    #                 ],
    #             }
    #         ]
    #         for _ in range(B)
    #     ]
    #     texts = [
    #         self.processor.apply_chat_template(
    #             conv, tokenize=False, add_generation_prompt=True
    #         ) + action_slots
    #         for conv in conversations
    #     ]
    #     inputs = self.processor(
    #         text=texts,
    #         images=[images_np[i] for i in range(B)],
    #         return_tensors="pt",
    #         padding=True,
    #     )
    #     return {k: v.to(device) for k, v in inputs.items()}
    def build_inputs(self, batch, device, image_mean, image_std, horizon=10):
        B = batch["pixel_values"].shape[0]
        T = batch["pixel_values"].shape[1]  # 16 frames

        # Undo VideoMAE norm for all frames → (B, T, H, W, 3) uint8
        frames = batch["pixel_values"].cpu()                          # (B, T, 3, H, W)
        frames = (frames * image_std.unsqueeze(1) + image_mean.unsqueeze(1)).clamp(0, 1)
        frames_np = frames.permute(0, 1, 3, 4, 2).mul(255).byte().numpy()  # (B, T, H, W, 3) 
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            for _ in range(B)
        ]
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            ) 
            for conv in conversations
        ]
        inputs = self.processor(
            text=texts,
            videos=[list(frames_np[i]) for i in range(B)],  # list of T frames per sample
            return_tensors="pt",
            padding=True,
        )
        
        return {k: v.to(device) for k, v in inputs.items()}

    def forward(self, horizon: int, **qwen_inputs) -> torch.Tensor:
        """
        Args:
            horizon:      number of action timesteps to predict (H)
            **qwen_inputs: output of build_inputs(), unpacked
        Returns:
            action_logits: (B, H, vocab_size)
        """
        outputs      = self.qwen(**qwen_inputs, output_hidden_states=True)
        hidden       = outputs.hidden_states[-1]          # (B, seq, hidden_dim)
        logits_full  = self.action_head(hidden)           # (B, seq, vocab_size)
        return logits_full[:, -horizon:, :]               # (B, H, vocab_size)

   
    def trainable_parameters(self):
        """Returns only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, output_dir: str):
        """Saves LoRA adapter + action head to output_dir."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.qwen.save_pretrained(f"{output_dir}/lora_adapter")
        torch.save(self.action_head.state_dict(), f"{output_dir}/action_head_weights.pt")
        print(f"Saved → {output_dir}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, vocab_size: int = 2048) -> "BaselineVLA":
        import os
        lora_path  = f"{checkpoint_dir}/lora_adapter"
        head_path  = f"{checkpoint_dir}/action_head_weights.pt"

        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"No LoRA adapter at {lora_path}")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"No action head at {head_path}")

        instance = cls(vocab_size=vocab_size)
        instance.qwen = PeftModel.from_pretrained(
            instance.qwen, lora_path, is_trainable=False
        )
        instance.action_head.load_state_dict(
            torch.load(head_path, map_location="cpu")
        )
        instance.eval()
        return instance