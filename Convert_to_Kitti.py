import cv2
import numpy as np
import os
import struct
import datetime as dt
import csv
import constants as c
start = dt.datetime.now()

# KITTI dataset folder structure
KITTI_PATH = "dataset/KITTI/"
os.makedirs(KITTI_PATH + "image_2", exist_ok=True)
os.makedirs(KITTI_PATH + "image_3", exist_ok=True)
os.makedirs(KITTI_PATH + "velodyne", exist_ok=True)
os.makedirs(KITTI_PATH + "calib", exist_ok=True)

def load_camera_calibration(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))

    return mtx_left, dist_left, mtx_right, dist_right

def load_processing_parameters(csv_file):
    params = {}
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            key = row[0]
            value = eval(row[1])
            params[key] = value

    return params["Common ROI (x, y, w, h)"], params["Common Image Size (w, h)"], params["Left ROI (x, y, w, h)"], params["Right ROI (x, y, w, h)"]

def undistort_and_crop(frame, mapx, mapy, roi, target_size):
    undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    cropped = undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted
    return cv2.resize(cropped, target_size)

def compute_focal_length(fov_deg, sensor_width_mm, image_width):
    fov_rad = np.radians(fov_deg)
    focal_length_px = (image_width / 2) / np.tan(fov_rad / 2)
    return focal_length_px

def save_bin(file_path, points):
    with open(file_path, 'wb') as f:
        for p in points:
            f.write(struct.pack('ffff', p[0], p[1], p[2], p[3]))

# Load calibration and processing parameters
mtx_left, dist_left, mtx_right, dist_right = load_camera_calibration("csv/camera_calibration_results.csv")
common_roi, common_size, roi_left, roi_right = load_processing_parameters("csv/processing_parameters_2.csv")

# Create remap matrices
h, w = 1080, 1920  # Known camera resolution
mapx_left, mapy_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, mtx_left, (w, h), cv2.CV_32FC1)
mapx_right, mapy_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, None, mtx_right, (w, h), cv2.CV_32FC1)

# Video capture
left_cap = cv2.VideoCapture("video/preprocessed/left.avi")
right_cap = cv2.VideoCapture("video/preprocessed/right.avi")

frame_id = 0

# Camera parameters
FOV = 130
SENSOR_WIDTH = 2.85
focal_length = compute_focal_length(FOV, SENSOR_WIDTH, common_size[0])
baseline = 0.12

while left_cap.isOpened() and right_cap.isOpened():
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    if not ret_left or not ret_right:
        break

    # Undistort, crop, and resize
    left_rect = undistort_and_crop(left_frame, mapx_left, mapy_left, roi_left, common_size)
    right_rect = undistort_and_crop(right_frame, mapx_right, mapy_right, roi_right, common_size)

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Save images
    left_img_path = f"{KITTI_PATH}/image_2/{frame_id:06d}.png"
    right_img_path = f"{KITTI_PATH}/image_3/{frame_id:06d}.png"
    cv2.imwrite(left_img_path, left_gray)
    cv2.imwrite(right_img_path, right_gray)

    # Disparity map
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Depth map
    depth = (focal_length * baseline) / (disparity + 1e-6)
    depth[depth > 80] = 80

    # Create point cloud
    h, w = depth.shape
    points = []
    for y in range(h):
        for x in range(w):
            z = depth[y, x]
            if z > 0:
                points.append([x, y, z, 1.0])

    # Save point cloud
    bin_path = f"{KITTI_PATH}/velodyne/{frame_id:06d}.bin"
    save_bin(bin_path, points)

    print(f"✅ Frame {frame_id} processed → PNG + BIN saved")
    frame_id += 1

left_cap.release()
right_cap.release()
end = dt.datetime.now()
print("✅ KITTI dataset conversion complete! Files saved in dataset/KITTI/")
print(f" Processing Time: {end - start}")
