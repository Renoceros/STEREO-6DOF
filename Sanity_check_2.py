from warnings import filters
import cv2
import threading

import numpy as np
import utils.stereo_utils as su
import config as c

def process_frame(frame, mapx_left, mapy_left, mapx_right, mapy_right, common_roi, common_size, output):
    """Processes and prepares both left and right frames using the common ROI."""
    frame = filters(frame)
    
    left_frame, right_frame = su.split_stereo_frame(frame)
    
    # Undistort, crop and resize using the common ROI
    left_processed = su.undistort_crop_resize(left_frame, mapx_left, mapy_left, common_roi, common_size)
    right_processed = su.undistort_crop_resize(right_frame, mapx_right, mapy_right, common_roi, common_size)

    output["left"] = left_processed
    output["right"] = right_processed

def filters(image):
    #GRAY
    image_gay = su.convert_to_grayscale(image)

    # Compute the gradient of the image using Sobel operator
    sobel_x = cv2.Sobel(image_gay, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(image_gay, cv2.CV_64F, 0, 1, ksize=1)

    # Calculate the magnitude of the gradient (the contrast)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the gradient to range 0-255
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert back to uint8 for visualization
    gradient_magnitude_normalized = np.uint8(gradient_magnitude_normalized)

    # Bump contrast by scaling pixel values
    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)

    # Apply contrast and brightness adjustment
    gradient_magnitude_normalized_contrast = cv2.convertScaleAbs(gradient_magnitude_normalized, alpha=alpha, beta=beta)



    return gradient_magnitude_normalized_contrast

def main():
    # Load calibration and precomputed parameters
    mtx_left, dist_left, mtx_right, dist_right = su.load_camera_calibration(c.calibration_csv)
    common_roi, common_size, _, _ = su.load_processing_parameters(c.processing_csv)  # Use common ROI, ignore left/right ROIs
    start, start_str = su.Current()
    print("Start Time : " + start_str)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    

    # Initialize undistort maps using stereo_utils
    mapx_left, mapy_left = su.create_undistort_map(mtx_left, dist_left, (640, 480))
    mapx_right, mapy_right = su.create_undistort_map(mtx_right, dist_right, (640, 480))
    su.Ding()
    end, end_str = su.Current()
    print("End Time : " + end_str)
    print("Duration : " + str(end - start))
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
