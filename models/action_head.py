import torch
import torch.nn as nn

class DiscreteActionHead(nn.Module):
    def __init__(self, hidden_dim=2048, vocab_size=2048):
        """
        Maps the Language Model's hidden dimension to a discrete action codebook representation.
        """
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        """
        Args:
            x (Tensor): Qwen2.5-VL-3B hidden states (batch_size, sequence_length, hidden_dim)
            
        Returns:
            Tensor: Action logits (batch_size, sequence_length, vocab_size)
        """
        return self.head(x)
