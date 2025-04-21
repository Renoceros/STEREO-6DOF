import cv2
import numpy as np
import json
from pupil_apriltags import Detector
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
CHESSBOARD_SIZE = (18, 12)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = 0  # Change if need be

# ===== Functions =====

def Capture_Stereo_Chessboard(chessboard_size):
    cap, orig_width, orig_height = su.OpenCam(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Capture] Failed to read from camera.")
            continue

        gray = su.Convert_to_Grayscale(frame)
        gray = cv2.equalizeHist(gray)
        left_img, right_img = su.Split_Stereo_Frame(gray)

        ret_l, corners_l = cv2.findChessboardCorners(left_img, chessboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(right_img, chessboard_size, None)

        display = frame.copy()
        if ret_l:
            cv2.drawChessboardCorners(display[:, :640], chessboard_size, corners_l, ret_l)
        if ret_r:
            cv2.drawChessboardCorners(display[:, 640:], chessboard_size, corners_r, ret_r)

        cv2.imshow("Stereo Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("[Capture] Exit requested.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == 32:
            if ret_l and ret_r:
                cap.release()
                cv2.destroyAllWindows()
                print("[Capture] Chessboard found in both views.")
                return corners_l, corners_r, left_img.shape[::-1], left_img, right_img
            else:
                print("[Capture] RETRY: Chessboard not found in both views.")


def Detect_AprilTags(left_img, right_img):
    detector = Detector()

    # Detect AprilTags in both images
    tags_left = detector.detect(left_img)
    tags_right = detector.detect(right_img)

    return tags_left, tags_right


def Triangulate_Points(tags_left, tags_right, Q_matrix):
    # For triangulation, we assume both left and right tags are aligned by the same ID.
    points_3d = []
    for tag_l, tag_r in zip(tags_left, tags_right):
        if tag_l.tag_id == tag_r.tag_id:  # Match by tag ID
            # Project points into 3D space using Q matrix (Assumes you have corresponding points from both sides)
            pt_left = np.array([tag_l.center[0], tag_l.center[1], 1])  # u_l, v_l, 1
            pt_right = np.array([tag_r.center[0], tag_r.center[1], 1])  # u_r, v_r, 1
            point_3d = np.dot(Q_matrix, np.array([pt_left[0], pt_left[1], pt_right[0], pt_right[1]]))
            points_3d.append(point_3d[:3])  # Only take X, Y, Z

    return np.array(points_3d)


def Load_Calibration_Data():
    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
    return calib_data


def Show_3D_Points(points_3d):
    # Simple point cloud visualization
    for point in points_3d:
        print(f"Point: {point}")
    # Visualize 3D in any way you'd like, e.g., using Open3D or Matplotlib


def Stereo_Test():
    calib_data = Load_Calibration_Data()
    Q_matrix = np.array(calib_data["Q"])

    cap, orig_width, orig_height = su.OpenCam(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Stereo_Test] Failed to read from camera.")
            continue

        gray = su.Convert_to_Grayscale(frame)
        gray = cv2.equalizeHist(gray)
        left_img, right_img = su.Split_Stereo_Frame(gray)

        # Detect AprilTags in both images
        tags_left, tags_right = Detect_AprilTags(left_img, right_img)

        if len(tags_left) > 0 and len(tags_right) > 0:
            # Triangulate 3D points based on the tag correspondences
            points_3d = Triangulate_Points(tags_left, tags_right, Q_matrix)

            # Display points
            Show_3D_Points(points_3d)

            # You can also draw the tags on the images for visual feedback
            for tag in tags_left:
                left_img = cv2.drawContours(left_img, [tag.corners.astype(np.int32)], -1, (0, 255, 0), 2)
            for tag in tags_right:
                right_img = cv2.drawContours(right_img, [tag.corners.astype(np.int32)], -1, (0, 255, 0), 2)

            # Combine and show both images
            combined = np.hstack((left_img, right_img))
            cv2.imshow("Stereo Images", combined)

        # Key to break the loop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    Stereo_Test()
