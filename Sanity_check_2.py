#Sanity_check_2.py
import cv2
import numpy as np
import csv
import threading

def load_camera_calibration(csv_file):
    """Loads camera calibration data from CSV."""
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))
    
    return mtx_left, dist_left, mtx_right, dist_right

def load_processing_parameters(csv_file):
    """Loads precomputed processing parameters from CSV."""
    params = {}
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            key = row[0]
            value = eval(row[1])  # Convert string to tuple
            params[key] = value

    return params["Common ROI (x, y, w, h)"], params["Common Image Size (w, h)"], params["Left ROI (x, y, w, h)"], params["Right ROI (x, y, w, h)"]

def undistort_and_crop(frame, mapx, mapy, roi, target_size):
    """Undistorts, crops, and resizes an image."""
    undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    cropped = undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted
    return cv2.resize(cropped, target_size)

def process_frame(frame, mapx_left, mapy_left, roi_left, mapx_right, mapy_right, roi_right, common_size, output):
    """Processes and prepares both left and right frames."""
    left_frame = frame[:, :1920]
    right_frame = frame[:, 1920:]

    left_undistorted = undistort_and_crop(left_frame, mapx_left, mapy_left, roi_left, common_size)
    right_undistorted = undistort_and_crop(right_frame, mapx_right, mapy_right, roi_right, common_size)

    output["left"] = cv2.cvtColor(left_undistorted, cv2.COLOR_BGR2GRAY)
    output["right"] = cv2.cvtColor(right_undistorted, cv2.COLOR_BGR2GRAY)

def main():
    # Load calibration and precomputed parameters
    mtx_left, dist_left, mtx_right, dist_right = load_camera_calibration("csv/camera_calibration_results.csv")
    common_roi, common_size, roi_left, roi_right = load_processing_parameters("csv/processing_parameters.csv")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Initialize undistort maps
    h, w = 1080, 1920  # Known resolution
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, mtx_left, (w, h), cv2.CV_32FC1)
    mapx_right, mapy_right = cv2.initUndistortRectifyMap(mtx_right, dist_right, None, mtx_right, (w, h), cv2.CV_32FC1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break
        
        output = {}
        thread = threading.Thread(target=process_frame, args=(frame, mapx_left, mapy_left, roi_left, mapx_right, mapy_right, roi_right, common_size, output))
        thread.start()
        thread.join()
        
        if "left" in output and "right" in output:
            cv2.imshow("Undistorted Left", output["left"])
            cv2.imshow("Undistorted Right", output["right"])

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
