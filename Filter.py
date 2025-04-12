#Filtet.py
import cv2
import os


def filter_images():
    """ Selects the best images based on chessboard detection and renames them."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = (18, 12)  # Adjust according to your pattern

    temp_path = "calibration_images/temp"
    left_final, right_final = None, None
    img_total = os.listdir(temp_path)
    num_pair_img = len(img_total) // 2
    print(num_pair_img)

    for i in range(num_pair_img):
        left_img_path = f"{temp_path}/left_{i}.jpg"
        right_img_path = f"{temp_path}/right_{i}.jpg"
        
        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            continue
        
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        found_left, _ = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        found_right, _ = cv2.findChessboardCorners(gray_right, chessboard_size, None)
        
        if found_left and found_right:
            left_final = left_img
            right_final = right_img
            num_last_left = len(os.listdir("calibration_images/left"))
            num_last_right = len(os.listdir("calibration_images/right"))
            left_img_final_path = f"calibration_images/left/left_cal_{num_last_left}.jpg"
            right_img_final_path = f"calibration_images/right/right_cal_{num_last_right}.jpg"
            os.rename(left_img_path, left_img_final_path)
            os.rename(right_img_path, right_img_final_path)
            print("Saved final calibration images.")
            break

    if left_final is None or right_final is None:
        print("Warning: No valid chessboard images found. Try again.")
        return

if __name__ == "__main__":
    filter_images()