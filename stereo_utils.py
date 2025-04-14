# stereo_utils.py

import cv2
import numpy as np
import pandas as pd
import csv

def load_camera_calibration(csv_path):
    calibration_data = {}
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key = row[0]
            values = np.array(row[1:], dtype=np.float64)
            if 'mtx' in key:
                calibration_data[key] = values.reshape((3, 3))
            elif 'dist' in key:
                calibration_data[key] = values
    return calibration_data

def load_processing_parameters(csv_path):
    df = pd.read_csv(csv_path)
    def parse_roi(roi_str):
        return tuple(map(int, roi_str.strip('()').split(',')))
    roi_left = parse_roi(df['roi_left'][0])
    roi_right = parse_roi(df['roi_right'][0])
    common_roi = parse_roi(df['common_roi'][0])
    return roi_left, roi_right, common_roi

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
