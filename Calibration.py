# Calibration.py
import cv2
import numpy as np
import os
import csv
import utils.stereo_utils as su

def calibrate_camera_live():
    start = su.Current()
    print("Start Time : "+str(start))
    chessboard_size = (18, 12)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    os.makedirs("calibration_images/left", exist_ok=True)
    os.makedirs("calibration_images/right", exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    image_count = 0
    state = 'left'

    print("Press SPACE to capture, ESC to exit.")
    
    end = su.Current()
    print("End Time : "+str(end))
    print("Durration To show feed: "+str(end-start))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed error.")
            break

        height, width = frame.shape[:2]
        half_width = width // 2
        left = frame[:, :half_width]
        right = frame[:, half_width:]

        display = left if state == 'left' else right
        cv2.putText(display, f"Capture {state.upper()} image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to capture
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            print(f"{state.capitalize()} Image: Chessboard Found? {found}")

            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if state == 'left':
                    imgpoints_left.append(corners)
                    objpoints.append(objp)
                    filename = f"calibration_images/left/left_{image_count:02d}.jpg"
                    cv2.imwrite(filename, display)
                    state = 'right'
                else:
                    imgpoints_right.append(corners)
                    filename = f"calibration_images/right/right_{image_count:02d}.jpg"
                    cv2.imwrite(filename, display)
                    image_count += 1
                    state = 'left'

    cap.release()
    cv2.destroyAllWindows()

    if not objpoints:
        print("No valid chessboard pairs captured.")
        return

    print("Running calibration...")

    ret_left, mtx_left, dist_left, *_ = cv2.calibrateCamera(objpoints, imgpoints_left, gray.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, *_ = cv2.calibrateCamera(objpoints, imgpoints_right, gray.shape[::-1], None, None)

    print("Calibration complete.")
    print("Left Camera Matrix:\n", mtx_left)
    print("Right Camera Matrix:\n", mtx_right)

    os.makedirs("csv", exist_ok=True)
    with open("csv/camera_calibration_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Left Camera", "Right Camera"])
        writer.writerow(["Matrix", mtx_left.tolist(), mtx_right.tolist()])
        writer.writerow(["Distortion Coefficients", dist_left.tolist(), dist_right.tolist()])

    print("Saved calibration to csv/camera_calibration_results.csv")

if __name__ == "__main__":
    calibrate_camera_live()
