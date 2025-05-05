import cv2
import numpy as np
import os
from utility import stereo_utils as su
import json

def main():
    video_path = 0  # Path to your video file
    fw = 1280  # Width of the video frame
    fh = 480   # Height of the video frame

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    # Open the video file
    cap, w, h = su.OpenCam(video_path, fw, fh)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Load camera calibration from JSON
    with open('camera_calibration.json', 'r') as f:
        calib = json.load(f)
    
    mtx_left = np.array(calib['mtx_left'])
    dist_left = np.array(calib['dist_left'])
    mtx_right = np.array(calib['mtx_right'])
    dist_right = np.array(calib['dist_right'])
    R = np.array(calib['R'])  # Rotation matrix
    T = np.array(calib['T'])  # Translation vector

    # Stereo rectification
    rect_size = (fw, fh)  # Keep the original image size
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, rect_size, R, T, alpha=0
    )

    # Map the stereo images to the rectified space
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, rect_size, cv2.CV_32FC1)
    mapx_right, mapy_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, rect_size, cv2.CV_32FC1)

    # SGBM parameters for disparity computation
    window_size = 5
    min_disp = 0
    num_disp = 64  # must be divisible by 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    su.Ding()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Convert to grayscale for stereo processing
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Split the frame into left and right images
        left, right = su.Split_Stereo_Frame(frame)

        # Apply undistortion and rectification using the maps
        left_rectified = cv2.remap(left, mapx_left, mapy_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, mapx_right, mapy_right, cv2.INTER_LINEAR)

        # Compute disparity
        disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

        # Normalize disparity for visualization
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        # Stack images horizontally for display
        combined = cv2.hconcat([ 
            cv2.cvtColor(left_rectified, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(right_rectified, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
        ])

        cv2.imshow("Left | Right | Disparity", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
