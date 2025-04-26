# stereo_test_1.py
import cv2
import numpy as np
import json
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = "./video/raw/dataset.avi"

# ===== Functions =====

def Detect_ORB_Features(left_img, right_img):
    orb = cv2.ORB_create()
    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)
    return kp_l, des_l, kp_r, des_r

def Limit_Keypoint_Density(kp_list, des_list, bucket_size=20, max_per_bucket=3):
    h, w = FRAME_HEIGHT, FRAME_WIDTH
    buckets = {}

    for i, kp in enumerate(kp_list):
        x, y = int(kp.pt[0] // bucket_size), int(kp.pt[1] // bucket_size)
        if (x, y) not in buckets:
            buckets[(x, y)] = []
        buckets[(x, y)].append((kp, des_list[i]))

    limited_kp = []
    limited_des = []

    for bucket in buckets.values():
        bucket.sort(key=lambda x: -x[0].response)  # sort by strength
        for kp, des in bucket[:max_per_bucket]:
            limited_kp.append(kp)
            limited_des.append(des)

    return limited_kp, np.array(limited_des)


def Match_Features(des_l, des_r, kp_l, kp_r, y_thresh=2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = []
    for m in matches:
        pt_l = kp_l[m.queryIdx].pt
        pt_r = kp_r[m.trainIdx].pt
        if abs(pt_l[1] - pt_r[1]) <= y_thresh:  # Delta y threshold
            good_matches.append(m)
    return good_matches

def Triangulate_Points(kp_l, kp_r, matches, Q_matrix):
    points_3d = []
    for match in matches:
        pt_left = np.array([kp_l[match.queryIdx].pt[0], kp_l[match.queryIdx].pt[1], 1])
        pt_right = np.array([kp_r[match.trainIdx].pt[0], kp_r[match.trainIdx].pt[1], 1])

        # Triangulate using Q matrix
        point_3d = np.dot(Q_matrix, np.array([pt_left[0], pt_left[1], pt_right[0], pt_right[1]]))
        points_3d.append(point_3d[:3])
    return np.array(points_3d)

def Load_Calibration_Data():
    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
    return calib_data

def Show_3D_Points(points_3d):
    for point in points_3d:
        print(f"Point: {point}")

def Stereo_Test():
    calib_data = Load_Calibration_Data()
    Q_matrix = np.array(calib_data["Q"])

    # Open video file instead of camera
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[Stereo_Test] Error opening video file.")
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Stereo_Test] Video resolution: {orig_width}x{orig_height}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Stereo_Test] End of video, looping back to start.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Video is color, convert to gray first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            gray = frame.copy()

        # Equalize histogram to enhance contrast
        gray = cv2.equalizeHist(gray)
        gray = su.Edge(gray)
        # Split into left and right images
        mid = gray.shape[1] // 2
        left_img, right_img = gray[:, :mid], gray[:, mid:]

        kp_l, des_l, kp_r, des_r = Detect_ORB_Features(left_img, right_img)
        kp_l, des_l = Limit_Keypoint_Density(kp_l, des_l)
        kp_r, des_r = Limit_Keypoint_Density(kp_r, des_r)

        if des_l is None or des_r is None:
            print("[Stereo_Test] No descriptors found, skipping frame.")
            continue

        matches = Match_Features(des_l, des_r, kp_l, kp_r)

        if len(matches) > 0:
            points_3d = Triangulate_Points(kp_l, kp_r, matches, Q_matrix)
            Show_3D_Points(points_3d)

            # Draw matches
            img_matches = cv2.drawMatches(left_img, kp_l, right_img, kp_r, matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Feature Matches", img_matches)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Stereo_Test()
