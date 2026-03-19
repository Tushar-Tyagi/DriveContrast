import os
import glob
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

class ActionTokenizer:
    def __init__(self, vocab_size=2048):
        """
        Tokenizes continuous driving actions into a discrete codebook using K-Disk (K-means) clustering.
        """
        self.vocab_size = vocab_size
        self.kmeans = None
        self.centers = None

    def fit(self, data_dir, subset="Unconventional Dynamic Obstacles", split="train", sample_ratio=0.1):
        """
        Fit the action codebook from either (1) tar shards containing .q7_answer.txt, or
        (2) extracted .npy action files under data_dir (e.g. after scripts/extract_waymo_subset.py).
        """
        all_actions = []
        tar_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.tar"), recursive=True))
        npy_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))

        if tar_files:
            print(f"Fitting from {len(tar_files)} tar file(s) in {data_dir}...")
            import tarfile
            import re
            for file in tar_files:
                try:
                    with tarfile.open(file, "r") as tar:
                        for member in tar.getmembers():
                            if member.name.endswith(".q7_answer.txt"):
                                f = tar.extractfile(member)
                                if f:
                                    content = f.read().decode("utf-8")
                                    matches = re.findall(r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]", content)
                                    if matches:
                                        trajectory = np.array([[float(m[0]), float(m[1])] for m in matches])
                                        all_actions.append(trajectory)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        elif npy_files:
            print(f"Fitting from {len(npy_files)} extracted .npy file(s) in {data_dir}...")
            for path in npy_files:
                try:
                    traj = np.load(path)
                    if traj.ndim == 2 and traj.shape[0] > 0:
                        all_actions.append(traj)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        else:
            raise ValueError(
                f"No .tar or .npy files found under {data_dir}. "
                "Run download_waymo_subset.py and then either extract_waymo_subset.py (to fit from .npy) or keep tars (to fit from .q7_answer.txt)."
            )

        if not all_actions:
            raise ValueError("No valid trajectory actions found.")
        flattened_actions = np.concatenate(all_actions, axis=0)
        print(f"Fitting MiniBatchKMeans over {len(flattened_actions)} steps for {self.vocab_size} centers...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, random_state=42, batch_size=4096)
        self.kmeans.fit(flattened_actions)
        self.centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
        print("Clustering complete.")

    def save(self, output_path="action_centers.pt"):
        if self.centers is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        torch.save(self.centers, output_path)
        print(f"Saved vocabulary centers to {output_path}")

    def load(self, input_path="action_centers.pt"):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing {input_path}. Please fit and save the tokenizer first.")
        self.centers = torch.load(input_path)
        print(f"Loaded {self.vocab_size} cluster centers.")

    def encode(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        """
        Maps continuous actions to the nearest discrete token ID.
        Args: continuous_actions (B, H, A) or (H, A)
        Returns: discrete_tokens (B, H) or (H) 
        """
        if self.centers is None:
            raise ValueError("Centers not available. Call load() or fit().")
            
        original_shape = continuous_actions.shape[:-1]
        flat_actions = continuous_actions.view(-1, continuous_actions.shape[-1])
        
        # Calculate L2 distances to all centers
        distances = torch.cdist(flat_actions, self.centers.to(continuous_actions.device))
        
        # Find nearest center index
        tokens = torch.argmin(distances, dim=-1)
        return tokens.view(original_shape)

    def decode(self, discrete_tokens: torch.Tensor) -> torch.Tensor:
        """
        Maps discrete token IDs back to continuous action centroids.
        """
        if self.centers is None:
            raise ValueError("Centers not available. Call load() or fit().")
        # Ensure device match
        centers = self.centers.to(discrete_tokens.device)
        return centers[discrete_tokens]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to Impromptu directory")
    parser.add_argument("--output_file", type=str, default="data/action_centers.pt")
    args = parser.parse_args()
    
    tokenizer = ActionTokenizer(vocab_size=2048)
    tokenizer.fit(data_dir=args.dataset_dir, subset="Unconventional Dynamic Obstacles", sample_ratio=0.2)
    tokenizer.save(args.output_file)
