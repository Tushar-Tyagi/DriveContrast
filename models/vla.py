import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, VideoMAEModel
from peft import PeftModel

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
        self.vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        
        # 3. Projection Layer
        self.projector = ProjectionLayer(input_dim=self.vision_dim, output_dim=self.vlm_hidden_dim)
        
        # 4. Action Head
        self.action_head = DiscreteActionHead(hidden_dim=self.vlm_hidden_dim, vocab_size=vocab_size)

    def _get_base_decoder(self):
        from peft import PeftModel
        if isinstance(self.vlm, PeftModel):
            # PeftModel → LoraModel → CausalLM → VLModel
            return self.vlm.base_model.model.model
        else:
            # CausalLM → VLModel
            return self.vlm.model

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        if not self.use_vanilla:
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            vision_feats   = vision_outputs.last_hidden_state
        else:
            vision_feats = pixel_values

        projected_feats = self.projector(vision_feats).to(dtype=torch.bfloat16)

        vlm_outputs   = self._get_base_decoder()(
            inputs_embeds=projected_feats,
            attention_mask=attention_mask,
        )
        hidden_states = vlm_outputs.last_hidden_state
        action_logits = self.action_head(hidden_states)
        return action_logits

if __name__ == "__main__":
    # Test Instantiation (Will require internet to download configs)
    try:
        model = AutoVLA4D()
        print("Successfully instantiated AutoVLA4D.")
    except Exception as e:
        print(f"Failed to instantiate: {e}")
