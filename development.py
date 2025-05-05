import cv2
import numpy as np
import os
from utility import stereo_utils as su
import json

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
CHESSBOARD_SIZE = (18, 12)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
IMG_SOURCE = "video/raw/IMG_6104.jpg" #Change if need be

frame = cv2.imread(IMG_SOURCE)
gray = su.Convert_to_Grayscale(frame)
gray = cv2.equalizeHist(gray)
left_img, right_img = su.Split_Stereo_Frame(gray)

ret_l, corners_l = cv2.findChessboardCorners(left_img, CHESSBOARD_SIZE, None)
ret_r, corners_r = cv2.findChessboardCorners(right_img, CHESSBOARD_SIZE, None)

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objpoints = [objp]
imgpoints_left = [corners_l]
imgpoints_right = [corners_r]
image_size = left_img.shape[::-1]

flags = 0  # No CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


retval, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
objpoints, imgpoints_left, imgpoints_right,
None, None, None, None,
image_size,
criteria=criteria,
flags=flags
)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
mtx_l, dist_l, mtx_r, dist_r,
image_size, R, T, alpha=0
)

calib_data = {
"mtx_left": mtx_l.tolist(),
"dist_left": dist_l.tolist(),
"mtx_right": mtx_r.tolist(),
"dist_right": dist_r.tolist(),
"R": R.tolist(),
"T": T.tolist(),
"E": E.tolist(),
"F": F.tolist(),
"R1": R1.tolist(),
"R2": R2.tolist(),
"P1": P1.tolist(),
"P2": P2.tolist(),
"Q": Q.tolist(),
"image_size": image_size
}

with open(JSON_PATH, "w") as f:
        json.dump(calib_data, f, indent=4)

print("[Stereo_Calibrate] Calibration complete. Saved to", JSON_PATH)

