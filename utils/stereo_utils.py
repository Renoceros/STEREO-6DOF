# utils/stereo_utils.py
from tracemalloc import start
import cv2
from glm import e
import numpy as np
import pandas as pd
import csv
from datetime import datetime

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

def undistort_and_crop(frame, mapx, mapy, roi):
    undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]

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