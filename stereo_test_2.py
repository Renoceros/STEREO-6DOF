# stereo_test_2.py
import cv2
import numpy as np
import json
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = "./video/raw/dataset.avi"  # Path to your stereo AVI

# ===== Functions =====

def Load_Calibration_Data():
    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
    return calib_data

def Stereo_Test_2():
    calib_data = Load_Calibration_Data()
    Q_matrix = np.array(calib_data["Q"])

    cap, orig_width, orig_height = su.OpenCam(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)

    # StereoSGBM parameters
    min_disp = 0
    num_disp = 16*6  # Must be divisible by 16
    block_size = 7
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Stereo_Test_2] Failed to read frame.")
            break

        # Ensure grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Split into left and right images
        mid = FRAME_WIDTH // 2
        left_img, right_img = gray[:, :mid], gray[:, mid:]

        # Compute disparity map
        disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

        # Normalize for visualization
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q_matrix)

        # Mask out low disparity areas
        mask = disparity > min_disp
        output_points = points_3d[mask]

        # Optional: Show a few points for verification
        if output_points.shape[0] > 0:
            print(f"Sample 3D point: {output_points[0]} (total {len(output_points)} points)")

        # Display disparity
        cv2.imshow("Disparity Map", disp_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Stereo_Test_2()
