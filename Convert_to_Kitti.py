import cv2
import numpy as np
import os
import struct
import datetime as dt

start = dt.datetime.now()

# KITTI dataset folder structure
KITTI_PATH = "dataset/KITTI/"
os.makedirs(KITTI_PATH + "image_2", exist_ok=True)
os.makedirs(KITTI_PATH + "image_3", exist_ok=True)
os.makedirs(KITTI_PATH + "velodyne", exist_ok=True)
os.makedirs(KITTI_PATH + "calib", exist_ok=True)

# Load video files
left_cap = cv2.VideoCapture("video/preprocessed/left.avi")
right_cap = cv2.VideoCapture("video/preprocessed/right.avi")

# Get frame dimensions from video
frame_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
expected_size = (frame_width, frame_height)

frame_id = 0

def compute_focal_length(fov_deg, sensor_width_mm):
    """ Compute focal length (in pixels) from field of view & sensor width. """
    fov_rad = np.radians(fov_deg)
    focal_length_px = (expected_size[0] / 2) / np.tan(fov_rad / 2)
    return focal_length_px

# Camera parameters (adjust if needed)
FOV = 130  # Camera field of view in degrees
SENSOR_WIDTH = 2.85  # Sensor width in mm
focal_length = compute_focal_length(FOV, SENSOR_WIDTH)
baseline = 0.12  # Baseline in meters

def save_bin(file_path, points):
    """ Save a point cloud to a KITTI .bin file """
    with open(file_path, 'wb') as f:
        for p in points:
            f.write(struct.pack('ffff', p[0], p[1], p[2], p[3]))  # x, y, z, intensity

while left_cap.isOpened() and right_cap.isOpened():
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    if not ret_left or not ret_right:
        break  # End of video

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Resize to match detected video frame size
    left_gray = cv2.resize(left_gray, expected_size, interpolation=cv2.INTER_NEAREST)
    right_gray = cv2.resize(right_gray, expected_size, interpolation=cv2.INTER_NEAREST)

    # Save as PNG images
    left_img_path = f"{KITTI_PATH}/image_2/{frame_id:06d}.png"
    right_img_path = f"{KITTI_PATH}/image_3/{frame_id:06d}.png"
    cv2.imwrite(left_img_path, left_gray)
    cv2.imwrite(right_img_path, right_gray)

    # Generate disparity map (approximate depth)
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)

    # Convert disparity to pseudo-LiDAR (basic approximation)
    depth = (focal_length * baseline) / (disparity + 1e-6)  # Avoid division by zero
    depth[depth > 80] = 80  # Limit depth range (adjust as needed)

    # Create point cloud
    h, w = depth.shape
    points = []
    for y in range(h):
        for x in range(w):
            z = depth[y, x]
            if z > 0:
                points.append([x, y, z, 1.0])  # x, y, depth, intensity

    # Save point cloud as .bin
    bin_path = f"{KITTI_PATH}/velodyne/{frame_id:06d}.bin"
    save_bin(bin_path, points)

    print(f"✅ Frame {frame_id} processed → PNG + BIN saved")
    frame_id += 1

# Release videos
left_cap.release()
right_cap.release()
end = dt.datetime.now()
print("✅ KITTI dataset conversion complete! Files saved in dataset/KITTI/")
print(f" Processing Time: {end - start}")

# GING GANG GULI GULI WATCHA GINGGANG GU GING GANG GU 
