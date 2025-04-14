import cv2
import numpy as np
import csv

def load_camera_calibration(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))
    return mtx_left, dist_left, mtx_right, dist_right

def compute_preprocessing_params(mtx_left, dist_left, mtx_right, dist_right, resolution=(1920, 1080), scale=1.0):
    w, h = resolution
    newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w, h), scale, (w, h))
    newcameramtx_right, roi_right = cv2.getOptimalNewCameraMatrix(mtx_right, dist_right, (w, h), scale, (w, h))

    # Calculate the intersection of both ROIs
    common_x = max(roi_left[0], roi_right[0])
    common_y = max(roi_left[1], roi_right[1])
    common_w = min(roi_left[0] + roi_left[2], roi_right[0] + roi_right[2]) - common_x
    common_h = min(roi_left[1] + roi_left[3], roi_right[1] + roi_right[3]) - common_y
    common_roi = (common_x, common_y, common_w, common_h)
    common_size = (common_w, common_h)

    return roi_left, roi_right, common_roi, common_size

def save_processing_params(csv_file, common_roi, common_size, roi_left, roi_right):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Common ROI (x, y, w, h)", str(common_roi)])
        writer.writerow(["Common Image Size (w, h)", str(common_size)])
        writer.writerow(["Left ROI (x, y, w, h)", str(roi_left)])
        writer.writerow(["Right ROI (x, y, w, h)", str(roi_right)])

if __name__ == "__main__":
    calibration_file = "csv/camera_calibration_results.csv"
    output_file = "csv/processing_parameters_2.csv"
    resolution = (1920, 1080)

    mtx_left, dist_left, mtx_right, dist_right = load_camera_calibration(calibration_file)
    roi_left, roi_right, common_roi, common_size = compute_preprocessing_params(
        mtx_left, dist_left, mtx_right, dist_right, resolution)

    save_processing_params(output_file, common_roi, common_size, roi_left, roi_right)

    print("✅ New preprocessing_parameters_2.csv generated successfully!")
    print(f"Common ROI: {common_roi}")
    print(f"Common Image Size: {common_size}")
    print(f"Left ROI: {roi_left}")
    print(f"Right ROI: {roi_right}")
