# Calibration.py
import cv2
import numpy as np
import os
import csv
import utils.stereo_utils as su
import constants as c

def calibrate_camera_live():
    start, start_str = su.Current()
    print("Start Time : " + start_str)
    print("Loading calibration and processing parameters...")
    mtx_left, dist_left, mtx_right, dist_right = su.load_camera_calibration(c.calibration_csv)
    common_roi, common_image_size, _, _ = su.load_processing_parameters(c.processing_csv)

    print("Creating undistortion maps...")
    mapx_left, mapy_left = su.create_undistort_map(mtx_left, dist_left, (c.f_width // 2, c.f_height))
    mapx_right, mapy_right = su.create_undistort_map(mtx_right, dist_right, (c.f_width // 2, c.f_height))

    chessboard_size = (18, 12)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    os.makedirs("calibration_images/left", exist_ok=True)
    os.makedirs("calibration_images/right", exist_ok=True)

    cap, orig_width, orig_height = su.OpenCam(c.cam_src)
    su.Ding()
    image_count = 0
    state = 'left'

    print("Press SPACE to capture, ESC to exit.")

    end, end_str = su.Current()
    print("End Time : "+end_str)
    print("Durration : "+str(end-start))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed error.")
            break

        frame = su.convert_to_grayscale(frame) #<- now we try with no gray
        frame = cv2.equalizeHist(frame) #<- now we try with no equalizeHist
        #Let's try it like this for alpha channel consistency

        # frame = su.edge(frame)
#DOOOOOONT FORGET TO CHANGE THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        left_frame, right_frame = su.split_stereo_frame(frame)

        left_proc = su.undistort_crop_resize(left_frame, mapx_left, mapy_left, common_roi, common_image_size)
        right_proc = su.undistort_crop_resize(right_frame, mapx_right, mapy_right, common_roi, common_image_size)

        enhanced = left_proc if state == 'left' else right_proc

        # cv2.putText(enhanced, f"Capture {state.upper()} image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow("Calibration", enhanced)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to capture
            found, corners = cv2.findChessboardCorners(enhanced, chessboard_size, None)

            print(f"{state.capitalize()} Image: Chessboard Found? {found}")

            if found:
                corners = cv2.cornerSubPix(enhanced, corners, (11, 11), (-1, -1), criteria)

                if state == 'left':
                    imgpoints_left.append(corners)
                    objpoints.append(objp)
                    filename = f"calibration_images/left/left_{image_count:02d}.jpg"
                    cv2.imwrite(filename, enhanced)
                    state = 'right'
                else:
                    imgpoints_right.append(corners)
                    filename = f"calibration_images/right/right_{image_count:02d}.jpg"
                    cv2.imwrite(filename, enhanced)
                    image_count += 1
                    state = 'left'

    cap.release()
    cv2.destroyAllWindows()

    if not objpoints:
        print("No valid chessboard pairs captured.")
        return

    print("Running calibration...")

    ret_left, mtx_left, dist_left, *_ = cv2.calibrateCamera(objpoints, imgpoints_left, enhanced.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, *_ = cv2.calibrateCamera(objpoints, imgpoints_right, enhanced.shape[::-1], None, None)

    print("Calibration complete.")
    print("Left Camera Matrix:\n", mtx_left)
    print("Right Camera Matrix:\n", mtx_right)

    # Compute Q matrices from updated intrinsics
    baseline = 0.12  # in meters

    fx_left = mtx_left[0, 0]
    cx_left = mtx_left[0, 2]
    cy_left = mtx_left[1, 2]

    fx_right = mtx_right[0, 0]
    cx_right = mtx_right[0, 2]
    cy_right = mtx_right[1, 2]

    Q_left = np.array([
        [1, 0,   0,         -cx_left],
        [0, 1,   0,         -cy_left],
        [0, 0,   0,          fx_left],
        [0, 0, -1.0 / baseline, 0]
    ], dtype=np.float32)

    Q_right = np.array([
        [1, 0,   0,         -cx_right],
        [0, 1,   0,         -cy_right],
        [0, 0,   0,          fx_right],
        [0, 0,  1.0 / baseline, 0]
    ], dtype=np.float32)

    # Save to CSV
    os.makedirs("csv", exist_ok=True)
    with open("csv/camera_calibration_results_after_pproc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Left Camera", "Right Camera"])
        writer.writerow(["Matrix", mtx_left.tolist(), mtx_right.tolist()])
        writer.writerow(["Distortion Coefficients", dist_left.tolist(), dist_right.tolist()])
        writer.writerow(["Q Matrix", Q_left.tolist(), Q_right.tolist()])

    print("Saved calibration and Q matrices to csv/camera_calibration_results_after_pproc.csv")

if __name__ == "__main__":
    calibrate_camera_live()
