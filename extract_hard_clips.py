import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import csv
import os
from pathlib import Path
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as pb

# CONFIGURATION
VAL_TFRECORD_DIR = "/path/to/wod_e2e/validation/"
OUTPUT_DIR = "./hard_clips_output/"
TOP_N = 30

os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

def parse_proto(serialized_example):
    """Parses a serialized E2EDFrame."""
    frame = pb.E2EDFrame()
    frame.ParseFromString(serialized_example.numpy())
    return frame

def calculate_metrics(future_poses):
    """
    future_poses: shape (N, 2)
    Compute jerk, heading change, lateral deviation, total displacement.
    """
    # Assuming constant time interval dt=0.25s (from requirement)
    dt = 0.25

    # Velocity
    vel = np.diff(future_poses, axis=0) / dt
    # Acceleration
    acc = np.diff(vel, axis=0) / dt
    # Jerk
    jerk = np.diff(acc, axis=0) / dt

    max_jerk = np.max(np.linalg.norm(jerk, axis=1)) if len(jerk) > 0 else 0.0

    # Heading change
    headings = np.arctan2(vel[:, 1], vel[:, 0])
    heading_changes = np.abs(np.diff(headings))
    heading_changes = np.minimum(heading_changes, 2*np.pi - heading_changes)
    max_heading_change = np.max(np.degrees(heading_changes)) if len(heading_changes) > 0 else 0.0

    lateral_deviation = np.max(np.abs(future_poses[:, 1]))
    total_displacement = np.linalg.norm(future_poses[-1] - future_poses[0])

    return max_jerk, max_heading_change, lateral_deviation, total_displacement

def main():
    print(f"Scanning {VAL_TFRECORD_DIR}...")

    files = tf.io.matching_files(os.path.join(VAL_TFRECORD_DIR, "*.tfrecord"))
    dataset = tf.data.TFRecordDataset(files)

    hard_clips = []

    for i, raw_record in enumerate(dataset):
        try:
            frame = parse_proto(raw_record)

            # Extract future trajectory (EgoTrajectoryStates)
            # Assuming structure: EgoTrajectoryStates has future_states
            # You may need to verify field names with actual proto data
            future_states = np.array([[s.pos_x, s.pos_y] for s in frame.ego_trajectory_states.future_states])

            max_jerk, max_heading_change, lateral_deviation, total_displacement = calculate_metrics(future_states)

            # Simple criteria
            if max_jerk > 2.0 or max_heading_change > 30.0 or lateral_deviation > 1.5:
                # Add to list (placeholder for full logic)
                hard_clips.append({
                    "frame": frame,
                    "metrics": (max_jerk, max_heading_change, lateral_deviation, total_displacement)
                })

        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1} frames...")

    print(f"Found {len(hard_clips)} hard clips.")
    # ... further processing, visualization, and saving steps ...

if __name__ == "__main__":
    main()
