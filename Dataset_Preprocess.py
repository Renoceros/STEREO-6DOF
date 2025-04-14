# Dataset_Preprocess.py

import cv2
import numpy as np
import os
import utils.stereo_utils as su
import config

# === Configuration ===(Take from config.py)
video_path = config.video_path
calibration_csv = config.calibration_csv
processing_csv = config.processing_csv
base_output_dir = config.output_dir

existing_batches = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith('BATCH_')]
batch_num = len(existing_batches)
#AGASGGSAAHSDJASHGDASHFJDSHakhirnya bisa rapih
output_dir = os.path.join(base_output_dir, f'BATCH_{batch_num}')
os.makedirs(output_dir, exist_ok=True)

# === Load Calibration & Processing Parameters ===
calib = su.load_camera_calibration(calibration_csv)
roi_left, roi_right, common_roi = su.load_processing_parameters(processing_csv)

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
roi_left = su.scale_roi(roi_left, scale_x, scale_y)
roi_right = su.scale_roi(roi_right, scale_x, scale_y)
common_roi = su.scale_roi(common_roi, scale_x, scale_y)

# === Undistortion Maps ===
frame_dummy = np.zeros((orig_height, orig_width // 2, 3), dtype=np.uint8)
mapx_left, mapy_left = su.create_undistort_map(calib['mtx_left'], calib['dist_left'], frame_dummy.shape[1::-1])
mapx_right, mapy_right = su.create_undistort_map(calib['mtx_right'], calib['dist_right'], frame_dummy.shape[1::-1])

# === Output Video Writers ===
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_left = cv2.VideoWriter(os.path.join(output_dir, 'Left.avi'), fourcc, fps, (common_roi[2], common_roi[3]), isColor=False)
out_right = cv2.VideoWriter(os.path.join(output_dir, 'Right.avi'), fourcc, fps, (common_roi[2], common_roi[3]), isColor=False)

# === Process Video Frame-by-Frame ===
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    left_frame, right_frame = su.split_stereo_frame(frame)
    left_gray = su.convert_to_grayscale(left_frame)
    right_gray = su.convert_to_grayscale(right_frame)

    left_corrected = su.undistort_and_crop(left_gray, mapx_left, mapy_left, common_roi)
    right_corrected = su.undistort_and_crop(right_gray, mapx_right, mapy_right, common_roi)

    out_left.write(left_corrected)
    out_right.write(right_corrected)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out_left.release()
out_right.release()
print("Processing complete.")
