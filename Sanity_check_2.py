import cv2
import threading
import utils.stereo_utils as su
import config as c

def process_frame(frame, mapx_left, mapy_left, mapx_right, mapy_right, common_roi, common_size, output):
    """Processes and prepares both left and right frames using the common ROI."""
    left_frame, right_frame = su.split_stereo_frame(frame)
    
    # Convert to grayscale
    left_gray = su.convert_to_grayscale(left_frame)
    right_gray = su.convert_to_grayscale(right_frame)

    # Undistort, crop and resize using the common ROI
    left_processed = su.undistort_crop_resize(left_gray, mapx_left, mapy_left, common_roi, common_size)
    right_processed = su.undistort_crop_resize(right_gray, mapx_right, mapy_right, common_roi, common_size)

    output["left"] = left_processed
    output["right"] = right_processed

def main():
    # Load calibration and precomputed parameters
    mtx_left, dist_left, mtx_right, dist_right = su.load_camera_calibration(c.calibration_csv)
    common_roi, common_size, _, _ = su.load_processing_parameters(c.processing_csv)  # Use common ROI, ignore left/right ROIs

    cap = cv2.VideoCapture(c.vid_unprocessed)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize undistort maps using stereo_utils
    mapx_left, mapy_left = su.create_undistort_map(mtx_left, dist_left, (640, 480))
    mapx_right, mapy_right = su.create_undistort_map(mtx_right, dist_right, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        output = {}
        # Process frame in a separate thread
        thread = threading.Thread(target=process_frame, args=(
            frame, mapx_left, mapy_left, mapx_right, mapy_right, common_roi, common_size, output
        ))
        thread.start()
        thread.join()

        if "left" in output and "right" in output:
            cv2.imshow("Undistorted Left", output["left"])
            cv2.imshow("Undistorted Right", output["right"])

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
