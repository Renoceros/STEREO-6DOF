# Dataset_Preprocess.py

import cv2
import numpy as np
import os
from stereo_utils import (
    load_camera_calibration,
    load_processing_parameters,
    scale_roi,
    undistort_and_crop,
    split_stereo_frame,
    create_undistort_map,
    convert_to_grayscale
)

# === Configuration ===
video_path = 'Dataset.avi'
calibration_csv = 'csv/camera_calibration_results.csv'
processing_csv = 'csv/processing_parameters.csv'
output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)

# === Load Calibration & Processing Parameters ===
calib = load_camera_calibration(calibration_csv)
roi_left, roi_right, common_roi = load_processing_parameters(processing_csv)

# === Open Video ===
cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {orig_width}x{orig_height}")

# === Determine Base Resolution for Scaling ===
base_width = 3840
base_height = 1080
scale_x = orig_width / base_width
scale_y = orig_height / base_height
print(f"Scaling factors - X: {scale_x:.2f}, Y: {scale_y:.2f}")

# === Scale ROIs to Actual Video Resolution ===
roi_left = scale_roi(roi_left, scale_x, scale_y)
roi_right = scale_roi(roi_right, scale_x, scale_y)
common_roi = scale_roi(common_roi, scale_x, scale_y)

# === Undistortion Maps ===
frame_dummy = np.zeros((orig_height, orig_width // 2, 3), dtype=np.uint8)
mapx_left, mapy_left = create_undistort_map(calib['mtx_left'], calib['dist_left'], frame_dummy.shape[1::-1])
mapx_right, mapy_right = create_undistort_map(calib['mtx_right'], calib['dist_right'], frame_dummy.shape[1::-1])

# === Output Video Writers ===
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_left = cv2.VideoWriter(os.path.join(output_dir, 'left_output.avi'), fourcc, fps, (common_roi[2], common_roi[3]), isColor=False)
out_right = cv2.VideoWriter(os.path.join(output_dir, 'right_output.avi'), fourcc, fps, (common_roi[2], common_roi[3]), isColor=False)

# === Process Video Frame-by-Frame ===
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    left_frame, right_frame = split_stereo_frame(frame)
    left_gray = convert_to_grayscale(left_frame)
    right_gray = convert_to_grayscale(right_frame)

    left_corrected = undistort_and_crop(left_gray, mapx_left, mapy_left, common_roi)
    right_corrected = undistort_and_crop(right_gray, mapx_right, mapy_right, common_roi)

    out_left.write(left_corrected)
    out_right.write(right_corrected)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out_left.release()
out_right.release()
print("Processing complete.")
