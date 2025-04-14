# Dataset_Preprocess.py
import cv2
import numpy as np
import os
import utils.stereo_utils as su
import config

# === Configuration === (Take from config.py)
start, start_str = su.Current()
print("Start Time : "+start_str)

video_path = config.vid_unprocessed
calibration_csv = config.calibration_csv
processing_csv = config.processing_csv
base_output_dir = config.vid_preprocessed

existing_batches = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith('BATCH_')]
batch_num = len(existing_batches)
output_dir = os.path.join(base_output_dir, f'BATCH_{batch_num}')
os.makedirs(output_dir, exist_ok=True)

# === Load Calibration & Processing Parameters ===
mtx_left, dist_left, mtx_right, dist_right = su.load_camera_calibration(calibration_csv)
common_roi, common_image_size, _, _ = su.load_processing_parameters(processing_csv)  # Using common ROI only

# === Open Video ===
cap = cv2.VideoCapture(video_path)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {orig_width}x{orig_height}")

# === Sanity Check ===
if (orig_width, orig_height) != (1280, 480):
    raise ValueError("Unexpected video resolution. Expected 1280x480 for stereo 640x480 input.")

# === Undistortion Maps ===
mapx_left, mapy_left = su.create_undistort_map(mtx_left, dist_left, (640, 480))
mapx_right, mapy_right = su.create_undistort_map(mtx_right, dist_right, (640, 480))

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

    # Update to use stereo_utils function for undistort, crop, and resize using common ROI
    left_corrected = su.undistort_crop_resize(left_gray, mapx_left, mapy_left, common_roi, common_image_size)
    right_corrected = su.undistort_crop_resize(right_gray, mapx_right, mapy_right, common_roi, common_image_size)

    out_left.write(left_corrected)
    out_right.write(right_corrected)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out_left.release()
out_right.release()
print("Processing complete.")
end, end_str = su.Current()
print("End Time : "+end_str)
print("Duration : "+str(end-start))
