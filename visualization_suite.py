import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


def bev_to_pixel(actions, img_w=224, img_h=224):
    """
    Project BEV (Bird-Eye-View) trajectory coordinates onto the front-camera image
    using a pinhole camera model.

    Actions format (from Impromptu-VLA / Waymo):
        actions[:, 0] = X = forward distance in meters (positive = forward)
        actions[:, 1] = Y = lateral offset in meters  (positive = left)

    Camera model:
        Waymo front camera has ~82° horizontal FOV. For a 224px sensor width:
            f = (img_w / 2) / tan(HFOV / 2) ≈ 130 pixels
        Camera is mounted ~1.5m above the road plane.
    """
    forward = actions[:, 0]  # meters forward
    lateral = actions[:, 1]  # meters lateral (positive = left)

    # Clamp forward to avoid division by zero
    forward = np.clip(forward, 0.5, None)

    # Camera intrinsics (calibrated for Waymo front camera at 224x224)
    cx = img_w / 2.0      # principal point x (image center)
    cy = img_h * 0.42     # principal point y (approximate horizon line)
    f = 130.0              # focal length in pixels (~82° HFOV)
    cam_height = 1.5       # camera height above road plane in meters

    # Pinhole projection: BEV (forward, lateral) → image (u, v)
    # Horizontal: u = cx - lateral * f / forward
    #   (negative sign: positive lateral = left → lower u)
    u = cx - lateral * f / forward

    # Vertical: v = cy + camera_height * f / forward
    #   (objects on the ground plane appear below the horizon)
    v = cy + cam_height * f / forward

    return np.stack([u, v], axis=-1)


def visualize_trajectory(image, gt_actions, pred_actions, output_path):
    """
    Dual-panel visualization:
      Left:  Front-camera image (for scene context)
      Right: Bird-Eye-View (top-down) trajectory plot

    Args:
        image: np.ndarray (H, W, 3) or PIL.Image — the front-camera frame.
        gt_actions: np.ndarray (N, 2) — ground-truth BEV waypoints [forward_m, lateral_m].
        pred_actions: np.ndarray (N, 2) — predicted BEV waypoints [forward_m, lateral_m].
        output_path: str — file path for the saved visualization.
    """
    fig, (ax_cam, ax_bev) = plt.subplots(1, 2, figsize=(12, 5),
                                          gridspec_kw={'width_ratios': [1, 1]})

    # ── Left panel: camera image ──
    if isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"))
    else:
        img_arr = image
    
    # -------------------------------------------------------------
    # 2. Camera View (Left Panel)
    # -------------------------------------------------------------
    img_h, img_w = img_arr.shape[:2]
    ax_cam.imshow(img_arr)
    ax_cam.set_title("Front Camera", fontsize=12, fontweight='bold')
    ax_cam.axis('off')

    # -------------------------------------------------------------
    # 3. BEV Top-Down Trajectory (Right Panel)
    # -------------------------------------------------------------
    # BEV convention: X = forward (up in plot), Y = lateral (positive = left)
    # We plot lateral on x-axis and forward on y-axis for a natural top-down view
    gt_fwd = gt_actions[:, 0]
    gt_lat = -gt_actions[:, 1]  # Waymo Y is positive-left. Negate for intuitive plot.
    pred_fwd = pred_actions[:, 0]
    pred_lat = -pred_actions[:, 1]

    # Plot GT trajectory
    ax_bev.plot(gt_lat, gt_fwd, 'r-', linewidth=2, alpha=0.6)
    ax_bev.scatter(gt_lat, gt_fwd, c='red', s=50, label='Ground Truth',
                   alpha=0.9, edgecolors='darkred', linewidths=0.8, zorder=5)

    # Plot predicted trajectory
    ax_bev.plot(pred_lat, pred_fwd, 'b-', linewidth=2, alpha=0.6)
    ax_bev.scatter(pred_lat, pred_fwd, c='dodgerblue', s=50, label='Baseline Prediction',
                   alpha=0.9, edgecolors='navy', linewidths=0.8, zorder=5)

    # Mark ego vehicle at origin
    ax_bev.scatter([0], [0], c='green', s=120, marker='^', label='Ego Vehicle',
                   edgecolors='darkgreen', linewidths=1, zorder=6)

    # Formatting
    ax_bev.set_xlabel("Lateral offset (m)", fontsize=10)
    ax_bev.set_ylabel("Forward distance (m)", fontsize=10)
    ax_bev.set_title("Bird-Eye-View Trajectory", fontsize=12, fontweight='bold')
    ax_bev.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f"Visualization saved to {output_path}")


def run_visualization_suite(data_dir, output_dir):
    """
    Iterates through dataset files and creates visualizations.
    Expects paired {name}.mp4 and {name}.npy files in data_dir (recursively).
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_path = os.path.join(root, file)
                base_name = file.replace(".npy", "")
                mp4_path = os.path.join(root, f"{base_name}.mp4")

                if os.path.exists(mp4_path):
                    # Extract the first frame of the clip
                    cap = cv2.VideoCapture(mp4_path)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pred_actions = np.load(npy_path)  # (10, 2) — BEV in meters

                        # In a real pipeline, load actual GT from a separate source.
                        # For now, use the stored actions as GT and simulate a
                        # predicted trajectory with small perturbations.
                        gt_actions = pred_actions.copy()
                        # Add realistic noise scaled to action magnitude
                        noise_scale = np.clip(np.abs(pred_actions) * 0.05, 0.1, 2.0)
                        sim_pred = pred_actions + np.random.normal(0, noise_scale)

                        visualize_trajectory(
                            frame_rgb, gt_actions, sim_pred,
                            os.path.join(output_dir, f"{base_name}.png")
                        )


if __name__ == "__main__":
    data_dir = "/Users/tushar/Documents/Github/DriveContrast/data/Unconventional Dynamic Obstacles/val"
    output_dir = "/Users/tushar/Documents/Github/DriveContrast/visualizations"
    run_visualization_suite(data_dir, output_dir)
