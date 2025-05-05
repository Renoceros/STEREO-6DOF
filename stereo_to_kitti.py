# stereo_to_kitti.py
import os
import json
import cv2
import numpy as np

# ==== Global Constants ====
VIDEO_PATH = "./video/raw/dataset.mp4"
CALIB_PATH = "./camera_calibration_results.json"
OUTPUT_BASE = "./dataset/kitti_001"

# ==== Utility Functions ====

def load_calibration(json_path):
    with open(json_path, 'r') as f:
        calib = json.load(f)
    required_keys = ['mtx_left', 'dist_left', 'mtx_right', 'dist_right', 'R', 'T', 'Q']
    for key in required_keys:
        if key not in calib:
            raise ValueError(f"Calibration file missing key: {key}")
    return calib


def setup_kitti_folder(base_folder):
    os.makedirs(os.path.join(base_folder, 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(base_folder, 'image_3'), exist_ok=True)
    os.makedirs(os.path.join(base_folder, 'velodyne'), exist_ok=True)


def rectify_images(left_img, right_img, calib):
    h, w = left_img.shape[:2]
    mtx_left = np.array(calib['mtx_left'])
    dist_left = np.array(calib['dist_left'])
    mtx_right = np.array(calib['mtx_right'])
    dist_right = np.array(calib['dist_right'])
    R = np.array(calib['R'])
    T = np.array(calib['T'])

    # Rectify images ensuring output resolution remains at 640x480
    new_size = (640, 480)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, new_size, R, T, alpha=0
    )

    # Compute rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, new_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, new_size, cv2.CV_16SC2)

    # Apply remap to get rectified images
    left_rectified = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    return left_rectified, right_rectified, Q


def compute_disparity(left_rect, right_rect):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # must be divisible by 16
        blockSize=9,
        P1=8 * 1 * 9 ** 2,
        P2=32 * 1 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_rect, right_rect).astype(np.float32) / 16.0
    return disparity


def save_kitti_image(img, path):
    cv2.imwrite(path, img)


def save_kitti_velodyne(points_3d, path):
    mask = (points_3d[:, :, 2] > 0) & (np.isfinite(points_3d[:, :, 2]))
    points = points_3d[mask]
    # Add dummy intensity value (e.g., 1.0)
    points_with_intensity = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
    points_with_intensity.astype(np.float32).tofile(path)


def setup_kitti_folder(base_folder_root="./dataset"):
    # List all folders that start with 'kitti_'
    existing = [d for d in os.listdir(base_folder_root) if d.startswith('kitti_') and os.path.isdir(os.path.join(base_folder_root, d))]

    # Extract numbers
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name.split('_')[1]))
        except (IndexError, ValueError):
            pass  # Ignore folders not matching pattern

    next_number = (max(numbers) + 1) if numbers else 1
    new_folder_name = f'kitti_{next_number:03d}'
    new_folder_path = os.path.join(base_folder_root, new_folder_name)

    os.makedirs(os.path.join(new_folder_path, 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(new_folder_path, 'image_3'), exist_ok=True)
    os.makedirs(os.path.join(new_folder_path, 'velodyne'), exist_ok=True)

    return new_folder_path


# ==== Main Script ====

def main():
    calib = load_calibration(CALIB_PATH)
    output_folder = setup_kitti_folder()  # <-- now auto-generated
    print(f"Saving KITTI dataset to: {output_folder}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise IOError(f"Failed to open video: {VIDEO_PATH}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CHECKPOINT 1: Frame read

        # Check color or grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        # Check size
        if frame_gray.shape != (480, 1280):
            raise ValueError(f"Unexpected frame size: {frame_gray.shape}")

        # Split left/right
        left_raw = frame_gray[:, :640]
        right_raw = frame_gray[:, 640:]

        # Rectify
        left_rect, right_rect, Q_rectified = rectify_images(left_raw, right_raw, calib)

        # CHECKPOINT 2: Rectification done

        # Disparity Map
        disparity = compute_disparity(left_rect, right_rect)

        # CHECKPOINT 3: Disparity map calculated

        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q_rectified)

        # Save outputs
        save_kitti_image(left_rect, os.path.join(output_folder, 'image_2', f"{frame_idx:06d}.png"))
        save_kitti_image(right_rect, os.path.join(output_folder, 'image_3', f"{frame_idx:06d}.png"))
        save_kitti_velodyne(points_3d, os.path.join(output_folder, 'velodyne', f"{frame_idx:06d}.bin"))

        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    print(f"✅ Finished! Saved {frame_idx} frames to {output_folder}")


if __name__ == "__main__":
    main()
