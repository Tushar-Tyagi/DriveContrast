# This script will augment the current training data 
import glob
import os
import random
from pprint import pprint
import shutil
import cv2
import numpy as np

HOME = "/home/jeff/CS7643/DriveContrast"

def apply_noise_injection(video_path, dest_path, noise_std=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i < 5:
            out.write(frame)
            continue
        noise = np.random.normal(0, noise_std, frame.shape)
        new_frame = np.clip(frame + noise, 0, 255)
        out.write(new_frame.astype(np.uint8))
        
        i += 1
    cap.release()
    out.release()

def apply_cutouts(video_path, dest_path, num_cutouts=10, cutout_size=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Let the model at least see the first half frames
        if i >= 8:
            for _ in range(num_cutouts):
                x = random.randint(0, width - cutout_size)
                y = random.randint(0, height - cutout_size)

                frame[y: y + cutout_size, x: x + cutout_size] = 0
        
        out.write(frame)
        i += 1
    cap.release()
    out.release()

def apply_frame_drops(video_path, dest_path, num_drops=4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    i = 0
    indices = list(range(2, 16))
    drop_indices = set(random.sample(indices, min(num_drops, len(indices))))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Let the model at least see the first half frames
        if i in drop_indices:
            frame = np.zeros_like(frame)
        
        out.write(frame)
        i += 1
    cap.release()
    out.release()

def apply_combined(video_path, dest_path, noise_std=10, num_cutouts=5, cutout_size=10, num_drops=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    i = 0
    indices = list(range(5, 16))
    drop_indices = set(random.sample(indices, min(num_drops, len(indices))))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i >= 5:
            noise = np.random.normal(0, noise_std, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if i >= 8:
            for _ in range(num_cutouts):
                x = random.randint(0, width - cutout_size)
                y = random.randint(0, height - cutout_size)

                frame[y: y + cutout_size, x: x + cutout_size] = 0
        if i in drop_indices:
            frame = np.zeros_like(frame)
        
        out.write(frame)
        i += 1
    cap.release()
    out.release()


def split_samples(samples, seed=42):
    filenames = [os.path.basename(s) for s in samples]
    random.seed(seed)
    random.shuffle(filenames)

    split_size = len(filenames) // 5

    categories = [
        "Unaltered",
        "Noise_Injection",
        "Cutouts",
        "Frame_Drops",
        "Combination",
    ]

    splits = {}
    start = 0
    for i, category in enumerate(categories):
        start = i * split_size
        splits[category] = filenames[start: start + split_size]
    
    return splits

def augment_videos(split, split_type):
    base_out = f"{HOME}/data/AugmentedData/{split_type}"
    os.makedirs(base_out, exist_ok=True)
    for key, values in split.items():
        for value in values:
            file_name = os.path.splitext(value)[0]
            video = os.path.join(f"{HOME}/data/Unconventional Dynamic Obstacles/{split_type}", value)
            npy_file = os.path.join(f"{HOME}/data/Unconventional Dynamic Obstacles/{split_type}", file_name + ".npy")
            npy_dest    = os.path.join(base_out, f"{file_name}_{key}.npy")
            video_dest  = os.path.join(base_out, f"{file_name}_{key}.mp4")
            shutil.copy2(npy_file, npy_dest)
            print(video_dest)
            if key == "Unaltered":
                shutil.copy2(video, video_dest)
            elif key == "Noise_Injection":
                apply_noise_injection(video, video_dest)
            elif key == "Cutouts":
                apply_cutouts(video, video_dest)
            elif key == "Frame_Drops":
                apply_frame_drops(video, video_dest)
            elif key == "Combination":
                apply_combined(video, video_dest)

def main():
    print("Begin data augmentation")
    # Get total number of training samples and validation samples
    train_dir = f"{HOME}/data/Unconventional Dynamic Obstacles/train"
    val_dir   = f"{HOME}/data/Unconventional Dynamic Obstacles/val"

    train_samples = glob.glob(os.path.join(train_dir, "*.mp4"))
    val_samples   = glob.glob(os.path.join(val_dir,   "*.mp4"))

    # Randomly Split training data into 5 categories
    # Unaltered images
    # Noise injection
    # Cutouts
    # Frame drops
    # Combination of noise injection, cutouts, frame drops
    train_split = split_samples(train_samples)
    val_split = split_samples(val_samples)
    # pprint(train_split)
    augment_videos(train_split, "train")
    augment_videos(val_split, "val")

    print(f"Training samples:   {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

if __name__ == '__main__':
    main()