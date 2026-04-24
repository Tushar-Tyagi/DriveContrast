"""
Video-level augmentations for robustness evaluation.
Operates on VideoMAE-preprocessed tensors of shape (16, 3, 224, 224).
"""

import torch
import numpy as np


def apply_augmentation(pixel_values: torch.Tensor, augmentation: str) -> torch.Tensor:
    """
    Apply a named augmentation to a batch of video frames.

    Args:
        pixel_values: (B, 16, 3, 224, 224) or (16, 3, 224, 224)
        augmentation: one of 'unaltered', 'noise_injection', 'cutout', 'frame_drop'

    Returns:
        Augmented tensor of same shape.
    """
    if augmentation == "unaltered":
        return pixel_values
    elif augmentation == "noise_injection":
        return _noise_injection(pixel_values)
    elif augmentation == "cutout":
        return _cutout(pixel_values)
    elif augmentation == "frame_drop":
        return _frame_drop(pixel_values)
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


def _noise_injection(pixel_values: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Additive Gaussian noise on normalized pixel values.
    sigma=0.1 is roughly equivalent to ~25/255 in unnormalized space.
    """
    noise = torch.randn_like(pixel_values) * sigma
    return pixel_values + noise


def _cutout(pixel_values: torch.Tensor, num_patches: int = 4, patch_size: int = 32) -> torch.Tensor:
    """
    Random rectangular occlusions zeroed out.
    Applied uniformly to all frames in the clip for spatial consistency.
    """
    result = pixel_values.clone()
    has_batch = result.dim() == 5

    if has_batch:
        B, T, C, H, W = result.shape
    else:
        T, C, H, W = result.shape
        result = result.unsqueeze(0)
        B = 1

    for b in range(B):
        for _ in range(num_patches):
            # Random top-left corner
            y = np.random.randint(0, H - patch_size + 1)
            x = np.random.randint(0, W - patch_size + 1)
            # Zero out across all frames (simulates persistent occlusion like mud on lens)
            result[b, :, :, y:y + patch_size, x:x + patch_size] = 0.0

    if not has_batch:
        result = result.squeeze(0)
    return result


def _frame_drop(pixel_values: torch.Tensor, drop_ratio: float = 0.3) -> torch.Tensor:
    """
    Drop ~30% of frames and repeat nearest surviving frames to maintain 16-frame length.
    Simulates sensor dropout / transmission loss.
    """
    result = pixel_values.clone()
    has_batch = result.dim() == 5

    if has_batch:
        B, T, C, H, W = result.shape
    else:
        T, C, H, W = result.shape
        result = result.unsqueeze(0)
        B = 1

    for b in range(B):
        num_drop = max(1, int(T * drop_ratio))
        drop_indices = sorted(np.random.choice(T, num_drop, replace=False))

        # Build list of surviving frame indices
        surviving = [i for i in range(T) if i not in drop_indices]
        if len(surviving) == 0:
            surviving = [0]  # safety: keep at least one frame

        # For each dropped frame, copy the nearest surviving frame
        for d in drop_indices:
            nearest = min(surviving, key=lambda s: abs(s - d))
            result[b, d] = result[b, nearest]

    if not has_batch:
        result = result.squeeze(0)
    return result
