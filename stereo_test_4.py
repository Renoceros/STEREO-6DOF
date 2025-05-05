# stereo_test_4.py
import os
import cv2
import numpy as np
import json

import os
import cv2
import numpy as np
import json

# ==== Global Constants ====
VIDEO_PATH = "./video/raw/dataset.avi"
CALIB_PATH = "./camera_calibration_results.json"
OUTPUT_FOLDER = "./rect_test"  # Output folder for rectified images

# ==== Utility Functions ====

def load_calibration(json_path):
    with open(json_path, 'r') as f:
        calib = json.load(f)
    required_keys = [
        'mtx_left', 'dist_left', 'mtx_right', 'dist_right',
        'R1', 'R2', 'P1', 'P2', 'Q'
    ]
    for key in required_keys:
        if key not in calib:
            raise ValueError(f"Calibration file missing key: {key}")
    return calib


def draw_diagonal_lines(image):
    """Draw two diagonal red lines (from 0,0 to 640,480 and 640,0 to 0,480)"""
    image_copy = image.copy()
    # Draw the diagonals
    cv2.line(image_copy, (0, 0), (640, 480), (0, 0, 255), 2)  # Red line (0,0) to (640,480)
    cv2.line(image_copy, (640, 0), (0, 480), (0, 0, 255), 2)  # Red line (640,0) to (0,480)
    return image_copy

def rectify_images(left_img, right_img, calib):
    h, w = left_img.shape[:2]
    mtx_left = np.array(calib['mtx_left'])
    dist_left = np.array(calib['dist_left'])
    mtx_right = np.array(calib['mtx_right'])
    dist_right = np.array(calib['dist_right'])
    
    R1 = np.array(calib['R1'])
    R2 = np.array(calib['R2'])
    P1 = np.array(calib['P1'])
    P2 = np.array(calib['P2'])
    Q = np.array(calib['Q'])

    rect_size = (640, 480)

    # Precompute undistort rectify map using pre-calculated rectification matrices
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, rect_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, rect_size, cv2.CV_16SC2)

    # Rectify images
    left_rectified = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    return left_rectified, right_rectified


# ==== Main Script ====

def main():
    # Load calibration data
    calib = load_calibration(CALIB_PATH)

    # Create output folder if not exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise IOError(f"Failed to open video: {VIDEO_PATH}")

    frame_idx = 0
    max_frames = 10  # Process only the first 10 frames

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Split left and right frames
        left_raw = frame[:, :640]
        right_raw = frame[:, 640:]

        # Draw diagonal lines on the raw images before rectification
        left_raw_with_lines = draw_diagonal_lines(left_raw)
        right_raw_with_lines = draw_diagonal_lines(right_raw)

        # Save raw images with diagonal lines
        left_raw_output_path = os.path.join(OUTPUT_FOLDER, f"LR_{frame_idx+1:04d}.png")
        right_raw_output_path = os.path.join(OUTPUT_FOLDER, f"RR_{frame_idx+1:04d}.png")
        cv2.imwrite(left_raw_output_path, left_raw_with_lines)
        cv2.imwrite(right_raw_output_path, right_raw_with_lines)

        # Perform rectification
        left_rect, right_rect = rectify_images(left_raw_with_lines, right_raw_with_lines, calib)

        # Save rectified images
        left_output_path = os.path.join(OUTPUT_FOLDER, f"L_{frame_idx+1:04d}.png")
        right_output_path = os.path.join(OUTPUT_FOLDER, f"R_{frame_idx+1:04d}.png")
        cv2.imwrite(left_output_path, left_rect)
        cv2.imwrite(right_output_path, right_rect)

        print(f"Processed frame {frame_idx + 1}")

        frame_idx += 1

    cap.release()
    print("✅ Finished rectifying and saving 10 frames.")

if __name__ == "__main__":
    main()
