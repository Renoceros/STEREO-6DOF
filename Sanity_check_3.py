import cv2
import numpy as np
import open3d as o3d

# Path to KITTI dataset
KITTI_PATH = "dataset/KITTI/"

def load_point_cloud(bin_path):
    """ Load KITTI LiDAR point cloud from .bin file """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
    return points

def visualize_point_cloud(points):
    """ Visualize 3D point cloud using Open3D """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use x, y, z only
    o3d.visualization.draw_geometries([pcd])

def visualize_kitti_frame(frame_id):
    """ Visualize KITTI frame (left image, right image, and point cloud) """
    frame_str = f"{frame_id:06d}"  # Convert to KITTI 6-digit format

    # Load left and right images
    left_img_path = f"{KITTI_PATH}/image_2/{frame_str}.png"
    right_img_path = f"{KITTI_PATH}/image_3/{frame_str}.png"
    
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print(f"❌ Error: Missing images for frame {frame_str}")
        return

    # Show left and right images
    cv2.imshow(f"Left Image {frame_str}", left_img)
    cv2.imshow(f"Right Image {frame_str}", right_img)

    # Load and visualize LiDAR point cloud
    bin_path = f"{KITTI_PATH}/velodyne/{frame_str}.bin"
    try:
        points = load_point_cloud(bin_path)
        visualize_point_cloud(points)
    except FileNotFoundError:
        print(f"⚠️ Warning: No point cloud found for frame {frame_str}")

    # Wait for user to close images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================
# 🔹 Run visualization for a specific frame
# Change the number to view different frames
frame_number = 30  # Change this to view other frames
visualize_kitti_frame(frame_number)
