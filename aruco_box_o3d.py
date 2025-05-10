#aruco_box_o3d.py
import time
import numpy as np
import open3d as o3d
import cv2

from aruco_box_pose_estimation import CubePoseEstimator
from utility.stereo_utils import OpenCam, Split_Stereo_Frame

# === CONFIG ===
FPS = 15
CUBE_SIZE = 0.08  # in meters
VIDEO_PATH = "video/raw/ArUco.avi"

# === Open Stereo Video ===
cap, camW, camH = OpenCam(VIDEO_PATH)

# === Pose Estimator ===
estimator = CubePoseEstimator()

# === Open3D Visualizer Setup ===
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Aruco Cube Pose (Open3D)", width=960, height=720)

# Axis frame
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# Cube mesh
cube = o3d.geometry.TriangleMesh.create_box(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE)
cube.compute_vertex_normals()
cube.paint_uniform_color([1, 0, 0])
cube.translate([-CUBE_SIZE / 2, -CUBE_SIZE / 2, 0])  # center on base

vis.add_geometry(axis)
vis.add_geometry(cube)

# === OpenCV Display Setup ===
cv2.namedWindow("Stereo Raw", cv2.WINDOW_NORMAL)

# === Main Loop ===
try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        # --- 3D Pose Estimation ---
        rvec, tvec = estimator.process_frame(frame)
        
        if rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            
            # Reset cube pose and apply new one
            cube.translate(-cube.get_center())  # Reset
            cube.transform(T)
        
        # --- Open3D Render Update ---
        vis.update_geometry(cube)
        vis.poll_events()
        vis.update_renderer()
        
        # --- Show Left & Right Frames (Raw) ---
        left, right = Split_Stereo_Frame(frame)
        stacked = np.hstack([left, right])
        cv2.imshow("Stereo Raw", stacked)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        # --- FPS Control ---
        elapsed = time.time() - start_time
        delay = max(1.0 / FPS - elapsed, 0)
        time.sleep(delay)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
