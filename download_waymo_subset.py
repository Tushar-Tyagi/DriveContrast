from huggingface_hub import hf_hub_download
import os

repo_id = "aaaaaap/unstructed"
save_dir = "./data/waymo_subset"
os.makedirs(save_dir, exist_ok=True)

# Select 2 training shards and 1 validation shard. 
# This provides a mix of examples while keeping the total footprint under ~50-100GB
shards_to_download = [
    "waymo/waymo_train_shard_0000.tar",
    "waymo/waymo_train_shard_0001.tar",
    "waymo/waymo_train_shard_0002.tar",
    "waymo/waymo_train_shard_0003.tar",
    "waymo/waymo_train_shard_0004.tar",
    "waymo/waymo_train_shard_0005.tar",
    "waymo/waymo_train_shard_0006.tar",
    "waymo/waymo_train_shard_0007.tar",
    "waymo/waymo_train_shard_0008.tar",
    "waymo/waymo_train_shard_0009.tar",
    "waymo/waymo_val_shard_0000.tar",
    "waymo/waymo_val_shard_0001.tar",
    "waymo/waymo_val_shard_0002.tar",
    "waymo/waymo_val_shard_0003.tar"
]

for shard in shards_to_download:
    print(f"Downloading {shard}...")
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=shard,
        local_dir=save_dir
    )
print("Finished downloading Waymo subset.")
