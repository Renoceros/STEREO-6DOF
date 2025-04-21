import cv2
import numpy as np
import json
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = 0  # Change if need be

# ===== Functions =====

def Detect_ORB_Features(left_img, right_img):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors in both images
    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)

    return kp_l, des_l, kp_r, des_r


def Match_Features(des_l, des_r):
    # Use Brute Force Matcher (BFMatcher) for feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the left and right image
    matches = bf.match(des_l, des_r)

    # Sort them in order of distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def Triangulate_Points(kp_l, kp_r, matches, Q_matrix):
    # List to store 3D points
    points_3d = []
    
    for match in matches:
        # Get the corresponding points from both images
        pt_left = np.array([kp_l[match.queryIdx].pt[0], kp_l[match.queryIdx].pt[1], 1])  # u_l, v_l, 1
        pt_right = np.array([kp_r[match.trainIdx].pt[0], kp_r[match.trainIdx].pt[1], 1])  # u_r, v_r, 1
        
        # Triangulate the point using the Q matrix
        point_3d = np.dot(Q_matrix, np.array([pt_left[0], pt_left[1], pt_right[0], pt_right[1]]))
        points_3d.append(point_3d[:3])  # Only take X, Y, Z

    return np.array(points_3d)


def Load_Calibration_Data():
    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
    return calib_data


def Show_3D_Points(points_3d):
    # Simple point cloud visualization (you can replace with Open3D or Matplotlib)
    for point in points_3d:
        print(f"Point: {point}")
    # Visualize 3D in any way you'd like


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

        # Detect ORB features in both images
        kp_l, des_l, kp_r, des_r = Detect_ORB_Features(left_img, right_img)

        # Match the features between the left and right images
        matches = Match_Features(des_l, des_r)

        if len(matches) > 0:
            # Triangulate the 3D points based on the matched features
            points_3d = Triangulate_Points(kp_l, kp_r, matches, Q_matrix)

            # Display points
            Show_3D_Points(points_3d)

            # Draw the matches on the images for visual feedback
            img_matches = cv2.drawMatches(left_img, kp_l, right_img, kp_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the matched images
            cv2.imshow("Feature Matches", img_matches)

        # Key to break the loop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    Stereo_Test()
