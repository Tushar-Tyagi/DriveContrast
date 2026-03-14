import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import decord
from PIL import Image
from transformers import AutoImageProcessor

decord.bridge.set_bridge('torch')

class ImpromptuVLADataset(Dataset):
    def __init__(self, data_dir, subset="Unconventional Dynamic Obstacles", image_processor_name="MCG-NJU/videomae-base", num_frames=16, resolution=224, split="train"):
        """
        Dataloader for Impromptu VLA dataset. Loads 16-frame VideoMAE clips and their corresponding actions.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.num_frames = num_frames
        self.resolution = resolution
        self.split = split
        
        # We assume the directory contains video files and matching action logs
        # E.g., `data_dir/subset/split/video_0001.mp4`
        self.video_files = sorted(glob.glob(os.path.join(data_dir, subset, split, "*.mp4")))
        
        # Load VideoMAE image processor
        self.processor = AutoImageProcessor.from_pretrained(image_processor_name)

    def __len__(self):
        return len(self.video_files)
        
    def _get_action_for_video(self, video_path):
        """
        Mock action loader. Replace with actual Impromptu VLA action loading logic.
        Assuming actions are saved as NumPy arrays with same base name.
        Actions: (horizon, action_dim) e.g., (10, 3) for 5 seconds at 2Hz.
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        action_path = os.path.join(os.path.dirname(video_path), f"{base_name}.npy")
        
        if os.path.exists(action_path):
            actions = np.load(action_path) 
            return torch.tensor(actions, dtype=torch.float32)
        else:
            # Fallback mock actions if dataset isn't fully structured yet
            return torch.zeros((10, 3), dtype=torch.float32)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        
        # 1. Load video frames
        vr = decord.VideoReader(video_path, width=self.resolution, height=self.resolution)
        total_frames = len(vr)
        
        # 2. Sample 16 frames uniformly
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Pad by repeating the last frame if video is too short
            indices = np.arange(total_frames)
            pad = [total_frames - 1] * (self.num_frames - total_frames)
            indices = np.concatenate([indices, pad])
            
        frames = vr.get_batch(indices) # Shape: (16, H, W, 3)
        
        # Convert to list of PIL Images for the huggingface processor
        frames_pil = [Image.fromarray(frame.numpy()) for frame in frames]
        
        # 3. Process frames using VideoMAE processor (Normalizes and resizes)
        # Output shape is typically (num_frames, channels, height, width) -> (16, 3, 224, 224)
        processed_frames = self.processor(list(frames_pil), return_tensors="pt")["pixel_values"]
        
        # Flatten batch dimension injected by processor: (1, 16, 3, 224, 224) -> (16, 3, 224, 224)
        if processed_frames.dim() == 5:
            processed_frames = processed_frames.squeeze(0)
            
        # 4. Load corresponding continuous actions
        actions = self._get_action_for_video(video_path)
        
        return {
            "pixel_values": processed_frames,
            "continuous_actions": actions
        }

if __name__ == "__main__":
    # Test instantiation
    dataset = ImpromptuVLADataset(data_dir="./", subset="Unconventional Dynamic Obstacles", split="train")
    print(f"Dataset initialized with {len(dataset)} items.")
