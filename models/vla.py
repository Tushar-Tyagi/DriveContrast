import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, VideoMAEModel

from .projector import ProjectionLayer
from .action_head import DiscreteActionHead

class AutoVLA4D(nn.Module):
    def __init__(self, 
                 vlm_name="Qwen/Qwen2.5-VL-3B-Instruct", 
                 vision_name="MCG-NJU/videomae-base",
                 vocab_size=2048,
                 use_vanilla_backbone=False):
        """
        AutoVLA 4D model wrapper.
        Combines a Vision Encoder (VideoMAE-base by default) with Qwen2.5-VL-3B.
        """
        super().__init__()
        self.use_vanilla = use_vanilla_backbone
        
        # 1. Vision Encoder
        if self.use_vanilla:
            # Fallback to standard AutoVLA static backbones (e.g. SigLIP/DinoV2)
            print("Initializing vanilla AutoVLA static backbone (Placeholder)")
            self.vision_encoder = None
            self.vision_dim = 768
        else:
            # 4D VideoMAE backbone
            print(f"Initializing 4D Vision Encoder: {vision_name}")
            self.vision_encoder = VideoMAEModel.from_pretrained(vision_name)
            self.vision_dim = self.vision_encoder.config.hidden_size # 768 for base
            
        # 2. VLM Backbone (Qwen2.5-VL)
        print(f"Initializing VLM Backbone: {vlm_name}")
        # Note: In practice for training, this would be loaded via QLoRA. 
        # For the architecture definition we define the skeleton here.
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.vlm_hidden_dim = self.vlm.config.hidden_size # 2048 for 3B
        
        # 3. Projection Layer
        self.projector = ProjectionLayer(input_dim=self.vision_dim, output_dim=self.vlm_hidden_dim)
        
        # 4. Action Head
        self.action_head = DiscreteActionHead(hidden_dim=self.vlm_hidden_dim, vocab_size=vocab_size)

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        """
        Args:
            pixel_values (Tensor): (B, num_frames=16, C=3, H=224, W=224) 
            input_ids (Tensor): Optional tokenized prompts.
            
        Returns:
            Tensor: Action logits (B, seq_len, vocab_size)
        """
        if not self.use_vanilla:
            # VideoMAE expects (B, num_frames, C, H, W)
            # Output: (B, num_patches * time_patches, hidden_size)
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            vision_feats = vision_outputs.last_hidden_state # (B, seq_len_vision, 768)
        else:
            # Vanilla logic (e.g. process single images)
            vision_feats = pixel_values
            
        # Project vision features into LLM space
        projected_feats = self.projector(vision_feats) # (B, seq_len_vision, 2048)
        
        # Pass through the language model. 
        # Next-token prediction style forward pass using inputs_embeds
        # (In a real scenario, we'd interleave this with `input_ids` text embeddings)
        vlm_outputs = self.vlm.model(inputs_embeds=projected_feats, attention_mask=attention_mask)
        hidden_states = vlm_outputs.last_hidden_state # (B, seq_len, 2048)
        
        # Predict actions
        action_logits = self.action_head(hidden_states) # (B, seq_len, 2048)
        
        return action_logits

if __name__ == "__main__":
    # Test Instantiation (Will require internet to download configs)
    try:
        model = AutoVLA4D()
        print("Successfully instantiated AutoVLA4D.")
    except Exception as e:
        print(f"Failed to instantiate: {e}")
