#Pre-pro_param.py
import cv2
import numpy as np
import csv
import threading
import utils.stereo_utils as su
import constants as c


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

def undistort_and_crop_resize(frame, mapx, mapy, roi, target_size):
    undistorted = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    cropped = undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted
    return cv2.resize(cropped, target_size)

def process_frame(frame, mapx_left, mapy_left, roi_left, mapx_right, mapy_right, roi_right, common_size, output):
    left_frame, right_frame = su.split_stereo_frame(frame)
    left_undistorted = undistort_and_crop_resize(left_frame, mapx_left, mapy_left, roi_left, common_size)
    right_undistorted = undistort_and_crop_resize(right_frame, mapx_right, mapy_right, roi_right, common_size)
    output["left"] = su.convert_to_grayscale(left_undistorted)
    output["right"] = su.convert_to_grayscale(right_undistorted)

def save_processing_parameters(common_roi, common_size, roi_left, roi_right):
    with open(c.processing_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Common ROI (x, y, w, h)", common_roi])
        writer.writerow(["Common Image Size (w, h)", common_size])
        writer.writerow(["Left ROI (x, y, w, h)", roi_left])
        writer.writerow(["Right ROI (x, y, w, h)", roi_right])
    print("\n✅ Processing Parameters Saved to 'processing_parameters.csv'")
    print(f"🔹 Common ROI: {common_roi}")
    print(f"🔹 Common Image Size: {common_size}")
    print(f"🔹 Left ROI: {roi_left}")
    print(f"🔹 Right ROI: {roi_right}")

def main():
    start, start_str = su.Current()
    print("Start Time : "+start_str)

    mtx_left, dist_left, mtx_right, dist_right = su.load_camera_calibration(c.calibration_csv)
    cap = cv2.VideoCapture(c.vid_unprocessed)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.f_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.f_height)

    end, end_str = su.Current()
    print("End Time : "+end_str)
    print("Durration : "+str(end-start))

    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read initial frame.")
        return

    h, w = frame.shape[:2]
    mapx_left, mapy_left, roi_left, _ = init_undistort_map(mtx_left, dist_left, c.f_height, h)
    mapx_right, mapy_right, roi_right, _ = init_undistort_map(mtx_right, dist_right, c.f_height, h)
    
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

    # Save processing parameters
    save_processing_parameters(common_roi, common_size, roi_left, roi_right)

if __name__ == "__main__":
    main()