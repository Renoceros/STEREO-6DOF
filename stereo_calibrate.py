# stereo_calibrate.py
import os
import cv2
import numpy as np
import json
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
CHESSBOARD_SIZE = (18, 12)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = 0 #Change if need be

# ===== Functions =====
def Capture_Stereo_Chessboard(chessboard_size):
    
    cap, orig_width, orig_height = su.OpenCam(VIDEO_SOURCE,FRAME_WIDTH,FRAME_HEIGHT)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Capture] Failed to read from camera.")
            continue

        gray = su.Convert_to_Grayscale(frame)
        gray = cv2.equalizeHist(gray)
        left_img, right_img = su.Split_Stereo_Frame(gray)

        ret_l, corners_l = cv2.findChessboardCorners(left_img, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(right_img, chessboard_size, None)

        display = frame.copy()
        if ret_l:
            cv2.drawChessboardCorners(display[:, :640], chessboard_size, corners_l, ret_l)
        if ret_r:
            cv2.drawChessboardCorners(display[:, 640:], chessboard_size, corners_r, ret_r)

        cv2.imshow("Stereo Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("[Capture] Exit requested.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == 32:
            if ret_l and ret_r:
                cap.release()
                cv2.destroyAllWindows()
                print("[Capture] Chessboard found in both views.")
                return corners_l, corners_r, left_img.shape[::-1], left_img, right_img
            else:
                print("[Capture] RETRY: Chessboard not found in both views.")


def Stereo_Calibrate():
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    corners_l, corners_r, image_size, img_l, img_r = Capture_Stereo_Chessboard(CHESSBOARD_SIZE)

    objpoints = [objp]
    imgpoints_left = [corners_l]
    imgpoints_right = [corners_r]

    # Skip individual calibrateCamera, let stereoCalibrate compute intrinsics too
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

if __name__ == "__main__":
    Stereo_Calibrate()
