# stereo_test_2.py
import cv2
import numpy as np
import json
from utility import stereo_utils as su
import open3d as o3d

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = "./video/raw/dataset.avi"  # Path to your stereo AVI

# ===== Functions =====

def Load_Calibration_Data():
    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
    return calib_data

def Create_Open3D_PointCloud(points_3d, mask, color_img=None, voxel_size=0.01):
    # Only select valid points
    valid_points = points_3d[mask]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    if color_img is not None:
        valid_colors = color_img[mask]
        pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)

    # Downsample
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def Stereo_Test_2():
    calib_data = Load_Calibration_Data()
    Q_matrix = np.array(calib_data["Q"])

    cap, orig_width, orig_height = su.OpenCam(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)

    # StereoSGBM setup...
    window_size = 5
    min_disp = 0
    num_disp = 16*6  # Must be divisible by 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size ** 2,
        P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Stereo_Test_2] Failed to read frame.")
            break

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        mid = FRAME_WIDTH // 2
        left_img, right_img = gray[:, :mid], gray[:, mid:]

        disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        points_3d = cv2.reprojectImageTo3D(disparity, Q_matrix)

        mask = (disparity > min_disp) & (disparity < num_disp)

        # Create Open3D point cloud
        pcd = Create_Open3D_PointCloud(points_3d, mask)

        # Visualize
        o3d.visualization.draw_geometries([pcd])

        # Display 2D disparity as backup
        cv2.imshow("Disparity Map", disp_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Stereo_Test_2()