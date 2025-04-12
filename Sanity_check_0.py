#Sanity_check_0.py
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def load_calibration_data(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    
    mtx_left = np.array(eval(data[1][1]))
    mtx_right = np.array(eval(data[1][2]))
    dist_left = np.array(eval(data[2][1]))
    dist_right = np.array(eval(data[2][2]))
    
    return mtx_left, dist_left, mtx_right, dist_right

def undistort_images():
    left_folder = "calibration_images/left"
    right_folder = "calibration_images/right"
    
    mtx_left, dist_left, mtx_right, dist_right = load_calibration_data("camera_calibration_results.csv")
    
    left_images = sorted([os.path.join(left_folder, f) for f in os.listdir(left_folder) if f.endswith(".jpg")])
    right_images = sorted([os.path.join(right_folder, f) for f in os.listdir(right_folder) if f.endswith(".jpg")])
    
    if not left_images or not right_images:
        print("Error: No images found for undistortion.")
        return
    
    for left_img_path, right_img_path in zip(left_images, right_images):
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        undist_left = cv2.undistort(left_img, mtx_left, dist_left)
        undist_right = cv2.undistort(right_img, mtx_right, dist_right)
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Original Left Image")
        axs[0, 0].axis("off")
        
        axs[0, 1].imshow(cv2.cvtColor(undist_left, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("Undistorted Left Image")
        axs[0, 1].axis("off")
        
        axs[1, 0].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("Original Right Image")
        axs[1, 0].axis("off")
        
        axs[1, 1].imshow(cv2.cvtColor(undist_right, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("Undistorted Right Image")
        axs[1, 1].axis("off")
        
        plt.show()

if __name__ == "__main__":
    undistort_images()
