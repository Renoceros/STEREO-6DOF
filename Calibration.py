# calibration.py
import cv2
import numpy as np
import os
import csv

def calibrate_camera():
    chessboard_size = (18, 12)  # Adjust according to your pattern
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D points
    imgpoints_left = []  # 2D points for left camera
    imgpoints_right = []  # 2D points for right camera
    
    left_folder = "calibration_images/left"
    right_folder = "calibration_images/right"
    
    left_images = sorted([os.path.join(left_folder, f) for f in os.listdir(left_folder) if f.endswith(".jpg")])
    right_images = sorted([os.path.join(right_folder, f) for f in os.listdir(right_folder) if f.endswith(".jpg")])
    
    if not left_images or not right_images:
        print("Error: No calibration images found.")
        return
    
    for left_img_path, right_img_path in zip(left_images, right_images):
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        found_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        found_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
        
        if found_left and found_right:
            objpoints.append(objp)
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
    
    if not objpoints:
        print("Error: Chessboard not detected in any images.")
        return
    
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
    
    print("Calibration successful!")
    print("Left Camera Matrix:", mtx_left)
    print("Right Camera Matrix:", mtx_right)
    print("Left Distortion Coefficients:", dist_left)
    print("Right Distortion Coefficients:", dist_right)
    
    with open("csv/camera_calibration_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Left Camera", "Right Camera"])
        writer.writerow(["Matrix", mtx_left.tolist(), mtx_right.tolist()])
        writer.writerow(["Distortion Coefficients", dist_left.tolist(), dist_right.tolist()])
    
    print("Calibration results saved to camera_calibration_results.csv")
    
if __name__ == "__main__":
    calibrate_camera()
