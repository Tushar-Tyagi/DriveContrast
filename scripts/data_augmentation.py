# This script will augment the current training data 
import glob
import os

HOME = "/home/jeff/CS7643/DriveContrast"

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

    print(f"Training samples:   {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

if __name__ == '__main__':
    main()