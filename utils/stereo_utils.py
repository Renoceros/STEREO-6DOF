# utils/stereo_utils.py
import os
from tracemalloc import start
import winsound
import cv2
import numpy as np
import pandas as pd
import csv
import constants as c
from datetime import datetime
import json

def load_camera_calibration(csv_file):
    """Loads camera calibration data from CSV."""
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))
    
    return mtx_left, dist_left, mtx_right, dist_right

def load_processing_parameters(csv_file):
    """Loads precomputed processing parameters from CSV."""
    params = {}
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            key = row[0]
            value = eval(row[1])  # Convert string to tuple
            params[key] = value

    return params["Common ROI (x, y, w, h)"], params["Common Image Size (w, h)"], params["Left ROI (x, y, w, h)"], params["Right ROI (x, y, w, h)"]

def scale_roi(roi, scale_x, scale_y):
    x, y, w, h = roi
    return (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))

def undistort_crop_resize(gray_img, mapx, mapy, roi, target_size):
    """Undistorts, crops, and resizes a grayscale image."""
    undistorted = cv2.remap(gray_img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    cropped = undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted
    return cv2.resize(cropped, target_size)

def edge(image):
    #GRAY
    image_gay = convert_to_grayscale(image)

    # Compute the gradient of the image using Sobel operator
    sobel_x = cv2.Sobel(image_gay, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(image_gay, cv2.CV_64F, 0, 1, ksize=1)

    # Calculate the magnitude of the gradient (the contrast)
    g_m = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the gradient to range 0-255
    g_m_n = cv2.normalize(g_m, None, 0, 255, cv2.NORM_MINMAX)

    # Convert back to uint8 for visualization purposes ofc ofc
    g_m_n = np.uint8(g_m_n)

    # Bump contrast by scaling pixel values
    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)

    # Apply contrast and brightness adjustment
    g_m_n_c = cv2.convertScaleAbs(g_m_n, alpha=alpha, beta=beta)

    # Blend using a weighted addition (tweak alpha as needed)
    # x = 0.5
    # blended = cv2.addWeighted(image_gay, x, g_m_n_c, 1 - x, 0)

    return g_m_n_c


def split_stereo_frame(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    return frame[:, :mid], frame[:, mid:]

def create_undistort_map(mtx, dist, size):
    return cv2.initUndistortRectifyMap(mtx, dist, None, mtx, size, cv2.CV_32FC1)

def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def Current():
    current = datetime.now()
    current_str = datetime.now().strftime("%H:%M:%S")
    return current, current_str

def Ding():
    winsound.MessageBeep(winsound.MB_ICONASTERISK)
    return

def LoJ(key):
    """Load a variable from the JSON state file."""
    if not os.path.exists(c.json_path):
        print("[LoJ] State file does not exist.")
        return None
    with open(c.json_path, "r") as f:
        data = json.load(f)
    return data.get(key)

def UpJ(key, value):
    """Update a variable in the JSON state file."""
    data = {}
    if os.path.exists(c.json_path):
        with open(c.json_path, "r") as f:
            data = json.load(f)
    data[key] = value
    with open(c.json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[UpJ] {key} updated to {value}")

def get_Q_matrix(focal_length=804.0, cx=320.0, cy=240.0, baseline=0.12):
    """
    Generate the Q matrix (disparity-to-depth mapping) using stereo camera parameters.

    Parameters:
    - focal_length (float): focal length in pixels
    - cx (float): principal point x-coordinate
    - cy (float): principal point y-coordinate
    - baseline (float): baseline distance between cameras in meters

    Returns:
    - Q (np.ndarray): 4x4 disparity-to-depth mapping matrix
    """
    Q = np.array([
        [1, 0,   0,       -cx],
        [0, 1,   0,       -cy],
        [0, 0,   0,        focal_length],
        [0, 0, -1 / baseline, 0]
    ], dtype=np.float32)
    return Q
