import cv2
import numpy as np
import csv
import os
import time
import gc
import psutil

# Load calibration data
def load_camera_calibration(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))

    return mtx_left, dist_left, mtx_right, dist_right

# Load processing parameters
def load_processing_parameters(csv_file):
    params = {}
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            key = row[0]
            value = eval(row[1])  # Convert string to tuple
            params[key] = value

    return params["Common ROI (x, y, w, h)"], params["Common Image Size (w, h)"], params["Left ROI (x, y, w, h)"], params["Right ROI (x, y, w, h)"]

# Undistort, crop, and resize frames
def undistort_and_crop(frame, mapx, mapy, roi, target_size):
    frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    cropped = frame[y:y+h, x:x+w] if w > 0 and h > 0 else frame
    return cv2.resize(cropped, target_size)

# Process video
def preprocess_video(input_path, output_left, output_right, mtx_left, dist_left, mtx_right, dist_right, common_roi, common_size, roi_left, roi_right):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Couldn't open {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dynamically determine left and right width
    single_width = frame_width // 2

    # Use MJPG for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Define video writers
    left_writer = cv2.VideoWriter(output_left, fourcc, fps, common_size, isColor=False)
    right_writer = cv2.VideoWriter(output_right, fourcc, fps, common_size, isColor=False)

    # Generate undistort maps
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, mtx_left, (single_width, frame_height), cv2.CV_32FC1)
    mapx_right, mapy_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, None, mtx_right, (single_width, frame_height), cv2.CV_32FC1)

    print(f"📌 Processing {total_frames} frames...")

    frame_count = 0
    skipped_frames = 0  # Track skipped frames
    start_time = time.time()

    while frame_count < total_frames:  # ✅ Prevents reading beyond EOF
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Warning: Frame {frame_count} could not be read. Reason: {cap.get(cv2.CAP_PROP_POS_FRAMES)} / {total_frames}")
            skipped_frames += 1

            # Stop if too many consecutive frames are unreadable
            if skipped_frames > 50:
                print("❌ Too many unreadable frames. Stopping early.")
                break
            
            frame_count += 1
            continue  # Skip to the next frame

        left_frame = frame[:, :single_width]
        right_frame = frame[:, single_width:]

        left_processed = undistort_and_crop(left_frame, mapx_left, mapy_left, roi_left, common_size)
        right_processed = undistort_and_crop(right_frame, mapx_right, mapy_right, roi_right, common_size)

        # Convert to grayscale
        left_bw = cv2.cvtColor(left_processed, cv2.COLOR_BGR2GRAY)
        right_bw = cv2.cvtColor(right_processed, cv2.COLOR_BGR2GRAY)

        left_writer.write(left_bw)
        right_writer.write(right_bw)

        frame_count += 1

        # Logging progress
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / frame_count) * total_frames
            remaining_time = estimated_total_time - elapsed_time
            print(f"✅ Processed {frame_count}/{total_frames} frames... | ⏳ Time Left: {remaining_time:.2f}s | 💾 RAM: {psutil.virtual_memory().percent}% | 🚨 Skipped: {skipped_frames}")

        # Force memory cleanup every 500 frames
        if frame_count % 500 == 0:
            gc.collect()

    cap.release()
    left_writer.release()
    right_writer.release()
    print(f"✅ Preprocessing complete. Files saved to video/preprocessed/")
    print(f"🚨 Total skipped frames: {skipped_frames}")

def main():
    input_video = "video/raw/dataset2.avi"
    output_left = "video/preprocessed/left2.avi"
    output_right = "video/preprocessed/right2.avi"

    # Load parameters
    mtx_left, dist_left, mtx_right, dist_right = load_camera_calibration("csv/camera_calibration_results.csv")
    common_roi, common_size, roi_left, roi_right = load_processing_parameters("csv/processing_parameters.csv")

    # Ensure output directory exists
    os.makedirs("video/preprocessed", exist_ok=True)

    # Process video
    preprocess_video(input_video, output_left, output_right, mtx_left, dist_left, mtx_right, dist_right, common_roi, common_size, roi_left, roi_right)

if __name__ == "__main__":
    main()
