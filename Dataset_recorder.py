import cv2
import os
import time
import config as c
import utils.stereo_utils as su

def main():
    start, start_str = su.Current()
    print("Start Time : " + start_str)

    print("Loading calibration and processing parameters...")
    mtx_left, dist_left, mtx_right, dist_right = su.load_camera_calibration(c.calibration_csv)
    common_roi, common_image_size, _, _ = su.load_processing_parameters(c.processing_csv)

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.f_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.f_height)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Creating undistortion maps...")
    mapx_left, mapy_left = su.create_undistort_map(mtx_left, dist_left, (c.f_width // 2, c.f_height))
    mapx_right, mapy_right = su.create_undistort_map(mtx_right, dist_right, (c.f_width // 2, c.f_height))

    recording = False
    frame_id = 0
    output_dir = create_next_batch_dir()

    image2_dir = os.path.join(output_dir, "image_2")
    image3_dir = os.path.join(output_dir, "image_3")
    os.makedirs(image2_dir, exist_ok=True)
    os.makedirs(image3_dir, exist_ok=True)

    end, end_str = su.Current()
    print("End Time : " + end_str)
    print("Duration : " + str(end - start))

    print("Ready. Press SPACE to start/stop recording. ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read from camera.")
            continue

        left_frame, right_frame = su.split_stereo_frame(frame)
        left_gray = su.convert_to_grayscale(left_frame)
        right_gray = su.convert_to_grayscale(right_frame)

        left_proc = su.undistort_crop_resize(left_gray, mapx_left, mapy_left, common_roi, common_image_size)
        right_proc = su.undistort_crop_resize(right_gray, mapx_right, mapy_right, common_roi, common_image_size)

        # === Combine and Enhance Display ===
        display_frame = cv2.hconcat([left_proc, right_proc])
        display_frame = cv2.equalizeHist(display_frame)
        cv2.imshow("Stereo Recording", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("Exiting...")
            break

        if key == 32:  # SPACE key
            recording = not recording
            print("=== Recording Started ===" if recording else "=== Recording Stopped ===")

        if recording:
            # Lanczos Upscaling (2x)
            left_up = cv2.resize(left_proc, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
            right_up = cv2.resize(right_proc, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)

            # Save PNGs
            left_path = os.path.join(image2_dir, f"{frame_id:06d}.png")
            right_path = os.path.join(image3_dir, f"{frame_id:06d}.png")
            cv2.imwrite(left_path, left_up)
            cv2.imwrite(right_path, right_up)
            print(f"{frame_id:06d} saved")
            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    end, end_str = su.Current()
    print("End Time : " + end_str)
    print("Duration Total : " + str(end - start))


def create_next_batch_dir():
    base_dir = c.img_preprocessed
    batch_folders = [f for f in os.listdir(base_dir) if f.startswith("BATCH_") and os.path.isdir(os.path.join(base_dir, f))]
    batch_num = len(batch_folders)
    new_batch_dir = os.path.join(base_dir, f"BATCH_{batch_num}")
    os.makedirs(new_batch_dir, exist_ok=True)
    return new_batch_dir


if __name__ == "__main__":
    main()
