# capture.py
import cv2
import os
import numpy as np

def capture_images():
    # Camera Settings
    CAMERA_INDEX = 0  # Change if necessary
    FRAME_WIDTH = 3840
    FRAME_HEIGHT = 1080
    CHESSBOARD_SIZE = (19, 13)  # Adjust according to your calibration pattern 
    SQUARE_SIZE = 15  # mm (adjust based on actual square size)

    # Output folders
    TEMP_FOLDER = "calibration_images/temp"
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Create fixed-size windows
    cv2.namedWindow("Left Camera", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Right Camera", cv2.WINDOW_AUTOSIZE)

    image_count = 0
    capturing = True

    while capturing:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Split frame into left and right images
        left_frame = frame[:, :FRAME_WIDTH // 2]
        right_frame = frame[:, FRAME_WIDTH // 2:]

        # Show the images without resizing
        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)

        # Capture images when spacebar is pressed
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar
            left_path = os.path.join(TEMP_FOLDER, f"left_{image_count}.jpg")
            right_path = os.path.join(TEMP_FOLDER, f"right_{image_count}.jpg")
            cv2.imwrite(left_path, left_frame)
            cv2.imwrite(right_path, right_frame)
            print(f"Captured image pair {image_count}")
            image_count += 1

        elif key == 27:  # ESC to stop capturing
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
