# sanity_check_2.9.py

import cv2
import numpy as np
import os
import open3d as o3d  # Assuming you'll use Open3D for point cloud output
import constants as c
import utils.stereo_utils as su

# === Depth Map Algorithms ===

def compute_stereoBM(left_gray, right_gray):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

def compute_stereoSGBM(left_gray, right_gray):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

# === Convert disparity to point cloud ===

def disparity_to_pointcloud(disparity, left_img, algorithm_name, Q, output_dir='PCD'):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > disparity.min()
    points = points_3D[mask]
    colors = left_img[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{algorithm_name}.pcd")
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    print(f"[{algorithm_name}] Point cloud saved to {output_path}")


def display_pointclouds(pcd_path_1, pcd_path_2):
    # Load both PCD files
    pcd1 = o3d.io.read_point_cloud(pcd_path_1)
    pcd2 = o3d.io.read_point_cloud(pcd_path_2)

    # Shift second point cloud for side-by-side view
    pcd2.translate((2, 0, 0))  # Move along x-axis for visual separation

    # Color them differently if needed
    pcd1.paint_uniform_color([1, 0.706, 0])  # orange-ish
    pcd2.paint_uniform_color([0, 0.651, 0.929])  # blue-ish

    # Display both
    print(f"Displaying:\n - {pcd_path_1}\n - {pcd_path_2}")
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Point Cloud Comparison")

# === Main function ===

def main():
    # You can modify this block:
    batch_dir = "image/preprocessed/BATCH_21"
    image_id = 10
    path_left = os.path.join(batch_dir, "image_2", f"{image_id:06d}.png")
    path_right = os.path.join(batch_dir, "image_3", f"{image_id:06d}.png")

    left_img = cv2.imread(path_left)
    right_img = cv2.imread(path_right)

    if left_img is None or right_img is None:
        print("Error: Could not load left or right image.")
        return

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # You can get a baseline Q matrix from stereo calibration
    Q = su.get_Q_matrix()  # You must define this in stereo_utils.py

    # Apply depth algorithms
    disparity_BM = compute_stereoBM(left_gray, right_gray)
    disparity_SGBM = compute_stereoSGBM(left_gray, right_gray)

    disparity_to_pointcloud(disparity_BM, left_img, "StereoBM", Q)
    disparity_to_pointcloud(disparity_SGBM, left_img, "StereoSGBM", Q)
    su.Ding
    display_pointclouds("PCD/StereoSGBM.pcd","PCD/StereoBM.pcd")

if __name__ == "__main__":
    main()
