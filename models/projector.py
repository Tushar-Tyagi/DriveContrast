import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModel

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=768, output_dim=2048):
        """
        Two-Layer MLP mapping VideoMAE-base features to Qwen2.5-VL-3B hidden dimension space.
        Uses GELU activation as specified. P(e) = Linear(GELU(Linear(e)))
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Output from VideoMAE encoder (batch_size, sequence_length, input_dim)
        Returns:
            Tensor: Projected embeddings (batch_size, sequence_length, output_dim)
        """
        return self.proj(x)
