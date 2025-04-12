import cv2
import numpy as np
import csv
import threading

def load_calibration_data(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))
    
    return mtx_left, dist_left, mtx_right, dist_right

def init_undistort_map(mtx, dist, w, h):
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
    return mapx, mapy, roi, new_mtx

def get_common_roi(roi_left, roi_right):
    x1 = max(roi_left[0], roi_right[0])
    y1 = max(roi_left[1], roi_right[1])
    x2 = min(roi_left[0] + roi_left[2], roi_right[0] + roi_right[2])
    y2 = min(roi_left[1] + roi_left[3], roi_right[1] + roi_right[3])
    
    return (x1, y1, x2 - x1, y2 - y1)  # (x, y, width, height)

def undistort_and_crop(frame, mapx, mapy, roi, target_size):
    undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    cropped = undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted
    return cv2.resize(cropped, target_size)

def process_frame(frame, mapx_left, mapy_left, roi_left, mapx_right, mapy_right, roi_right, common_size, output):
    left_frame = frame[:, :1920]
    right_frame = frame[:, 1920:]

    left_undistorted = undistort_and_crop(left_frame, mapx_left, mapy_left, roi_left, common_size)
    right_undistorted = undistort_and_crop(right_frame, mapx_right, mapy_right, roi_right, common_size)

    output["left"] = cv2.cvtColor(left_undistorted, cv2.COLOR_BGR2GRAY)
    output["right"] = cv2.cvtColor(right_undistorted, cv2.COLOR_BGR2GRAY)

def main():
    mtx_left, dist_left, mtx_right, dist_right = load_calibration_data("camera_calibration_results.csv")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read initial frame.")
        return

    h, w = frame.shape[:2]
    mapx_left, mapy_left, roi_left, _ = init_undistort_map(mtx_left, dist_left, 1920, h)
    mapx_right, mapy_right, roi_right, _ = init_undistort_map(mtx_right, dist_right, 1920, h)
    
    common_roi = get_common_roi(roi_left, roi_right)
    common_size = (common_roi[2], common_roi[3])  # (width, height)

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
