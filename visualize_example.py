import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from visualization_suite import bev_to_pixel


def visualize_trajectory(image, actions, output_path):
    """
    Visualizes Ground Truth (red) and Planning (blue) trajectories on an image.

    Args:
        image (np.ndarray): The base image (expected (224, 224, 3)).
        actions (np.ndarray): Predicted actions (10, 2) in BEV meters [forward, lateral].
        output_path (str): File path to save the visualization.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the image
    ax.imshow(image)
    h, w = image.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # Project BEV actions to pixel space
    pred_px = bev_to_pixel(actions, img_w=w, img_h=h)

    # Simulate GT with small perturbation for demonstration
    noise_scale = np.clip(np.abs(actions) * 0.05, 0.1, 2.0)
    gt_actions = actions + np.random.normal(0, noise_scale)
    gt_px = bev_to_pixel(gt_actions, img_w=w, img_h=h)

    # Draw trajectory lines
    ax.plot(gt_px[:, 0], gt_px[:, 1], c='red', linewidth=1.5, alpha=0.5)
    ax.plot(pred_px[:, 0], pred_px[:, 1], c='dodgerblue', linewidth=1.5, alpha=0.5)

    # Draw waypoints
    ax.scatter(gt_px[:, 0], gt_px[:, 1], c='red', s=50, label='Ground Truth',
               alpha=0.8, edgecolors='white', linewidths=0.5, zorder=5)
    ax.scatter(pred_px[:, 0], pred_px[:, 1], c='dodgerblue', s=50, label='Planning',
               alpha=0.8, edgecolors='white', linewidths=0.5, zorder=5)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.7)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_path}")


# Example usage
# dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
# dummy_actions = np.array([
#     [5.5, -0.01], [10.9, -0.01], [16.2, -0.02], [21.3, -0.03], [26.3, -0.06],
#     [31.0, -0.11], [35.6, -0.18], [39.8, -0.29], [43.8, -0.42], [47.5, -0.56]
# ])
# visualize_trajectory(dummy_img, dummy_actions, "visualization_example.png")
