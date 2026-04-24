import os
import re
import glob
import tarfile
import io
import numpy as np
import cv2
import gc
from PIL import Image
from collections import defaultdict

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TAR_DIR = os.path.join(PROJECT_ROOT, "data", "waymo_subset", "waymo")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
SUBSET = "Unconventional Dynamic Obstacles"

FPS = 2
RESOLUTION = 224
CLIP_SIZE = 16
STRIDE = 8

def parse_q7_answer(text: str) -> np.ndarray:
    matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if not matches:
        return np.zeros((10, 2), dtype=np.float32)
    return np.array([[float(x), float(y)] for x, y in matches], dtype=np.float32)

def pad_or_truncate_actions(actions: np.ndarray, horizon: int = 10) -> np.ndarray:
    n = actions.shape[0]
    if n == 0:
        return np.zeros((horizon, 2), dtype=np.float32)
    if n >= horizon:
        return actions[:horizon]
    pad = np.tile(actions[-1:], (horizon - n, 1))
    return np.concatenate([actions, pad], axis=0)

def frames_to_mp4(frames: list, output_path: str, fps: int = 2, resolution: int = 224):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (resolution, resolution))

    for pil_img in frames:
        if pil_img.size != (resolution, resolution):
            pil_img = pil_img.resize((resolution, resolution), Image.BILINEAR)
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()

def extract_tar(tar_path: str) -> dict:
    samples = {}
    with tarfile.open(tar_path, "r") as tar:
        members_by_name = {m.name: m for m in tar.getmembers()}
        front_keys = [name for name in members_by_name if name.endswith(".camera_FRONT.png")]

        for front_name in front_keys:
            base = front_name.replace(".camera_FRONT.png", "")
            sample_id = os.path.basename(base)

            def read_member(suffix):
                key = f"{base}.{suffix}"
                m = members_by_name.get(key)
                if m is None:
                    return None
                f = tar.extractfile(m)
                return f.read() if f else None

            png_bytes = read_member("camera_FRONT.png")
            if png_bytes is None:
                continue
            front_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

            clip_bytes = read_member("clip_id.txt")
            clip_id = clip_bytes.decode("utf-8").strip() if clip_bytes else "unknown"

            idx_bytes = read_member("idx.txt")
            try:
                frame_idx = int(idx_bytes.decode("utf-8").strip()) if idx_bytes else 0
            except ValueError:
                frame_idx = 0

            cat_bytes = read_member("category.txt")
            category = cat_bytes.decode("utf-8").strip() if cat_bytes else ""

            q7_bytes = read_member("q7_answer.txt")
            if q7_bytes:
                raw_actions = parse_q7_answer(q7_bytes.decode("utf-8"))
                actions = pad_or_truncate_actions(raw_actions, horizon=10)
            else:
                actions = np.zeros((10, 2), dtype=np.float32)

            samples[sample_id] = {
                "clip_id": clip_id,
                "idx": frame_idx,
                "front_png": front_img,
                "actions": actions,
                "category": category,
            }

    return samples

def build_clips(all_samples: dict, clip_size: int = 16, stride: int = 8) -> list:
    by_clip = defaultdict(list)
    for _, data in all_samples.items():
        by_clip[data["clip_id"]].append(data)

    clips = []
    for clip_id, frames_data in by_clip.items():
        frames_data.sort(key=lambda x: x["idx"])
        n = len(frames_data)
        if n == 0: continue

        if n < clip_size:
            pad_needed = clip_size - n
            padded = frames_data + [frames_data[-1]] * pad_needed
            clips.append({
                "clip_id": clip_id,
                "clip_window_idx": 0,
                "frames": [d["front_png"] for d in padded],
                "actions": frames_data[-1]["actions"],
            })
        else:
            window_idx = 0
            for start in range(0, n - clip_size + 1, stride):
                window = frames_data[start:start + clip_size]
                clips.append({
                    "clip_id": clip_id,
                    "clip_window_idx": window_idx,
                    "frames": [d["front_png"] for d in window],
                    "actions": window[-1]["actions"],
                })
                window_idx += 1
    return clips

def write_split(split_name: str, split_clips: list):
    out_dir = os.path.join(OUTPUT_DIR, SUBSET, split_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {len(split_clips)} {split_name} clips ...")

    for clip in split_clips:
        safe_clip_id = re.sub(r"[^\w\-]", "_", clip["clip_id"])
        base_name = f"clip_{safe_clip_id}_{clip['clip_window_idx']:04d}"
        frames_to_mp4(clip["frames"], os.path.join(out_dir, f"{base_name}.mp4"), fps=FPS, resolution=RESOLUTION)
        np.save(os.path.join(out_dir, f"{base_name}.npy"), clip["actions"].astype(np.float32))

def collect_split_shards(tar_dir: str):
    return sorted(glob.glob(os.path.join(tar_dir, "waymo_train_shard_*.tar"))), sorted(glob.glob(os.path.join(tar_dir, "waymo_val_shard_*.tar")))

def load_samples_from_shards(tar_files: list, split_name: str) -> dict:
    all_samples = {}
    for i, tar_path in enumerate(tar_files):
        print(f"Extracting {split_name} {i+1}/{len(tar_files)}")
        shard_samples = extract_tar(tar_path)
        for sample_id, sample in shard_samples.items():
            all_samples[f"{os.path.basename(tar_path)}__{sample_id}"] = sample
    return all_samples

def main():
    train_tars, val_tars = collect_split_shards(TAR_DIR)
    # train_samples = load_samples_from_shards(train_tars, "train")
    # write_split("train", build_clips(train_samples))
    val_samples = load_samples_from_shards(val_tars, "val")
    write_split("val", build_clips(val_samples))

if __name__ == "__main__":
    main()
