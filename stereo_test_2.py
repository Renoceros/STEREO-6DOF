# stereo_test_2.py
import cv2
import numpy as np
import json
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
VIDEO_SOURCE = 0
MATCH_EVERY_N_FRAMES = 10
EPIPOLAR_THRESH = 2.0  # pixels
ANGLE_THRESH = 15.0    # degrees

# ===== Functions =====
def Load_Calibration(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    Q = np.array(data['Q'])
    return Q

def Filter_Matches_By_Epipolar_And_Angle(kp1, kp2, matches, epipolar_thresh, angle_thresh):
    filtered = []
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        angle1 = kp1[m.queryIdx].angle
        angle2 = kp2[m.trainIdx].angle
        if abs(pt1[1] - pt2[1]) < epipolar_thresh and abs(angle1 - angle2) < angle_thresh:
            filtered.append((pt1, pt2))
    return filtered

def Triangulate_Points(correspondences, Q):
    points_3d = []
    for (pt_l, pt_r) in correspondences:
        disparity = pt_l[0] - pt_r[0]
        if disparity == 0:
            continue
        point_4d = np.dot(Q, np.array([pt_l[0], pt_l[1], disparity, 1.0]))
        point_3d = point_4d[:3] / point_4d[3]
        points_3d.append(point_3d)
    return np.array(points_3d)

def Detect_And_Match(left_img, right_img):
    orb = cv2.ORB_create(2000)
    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
    return kp_l, kp_r, matches

def Track_Keypoints(prev_img, next_img, prev_pts):
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, None)
    good_prev = prev_pts[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]
    return good_prev, good_next

def Run_Stereo_Tracking():
    Q = Load_Calibration(JSON_PATH)
    cap, _, _ = su.OpenCam(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)
    frame_count = 0
    prev_gray_l, prev_gray_r = None, None
    tracked_pts_l, tracked_pts_r = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = su.Convert_to_Grayscale(frame)
        left_img, right_img = su.Split_Stereo_Frame(gray)

        if frame_count % MATCH_EVERY_N_FRAMES == 0 or tracked_pts_l is None:
            kp_l, kp_r, matches = Detect_And_Match(left_img, right_img)
            correspondences = Filter_Matches_By_Epipolar_And_Angle(kp_l, kp_r, matches, EPIPOLAR_THRESH, ANGLE_THRESH)

            if not correspondences:
                frame_count += 1
                continue

            tracked_pts_l = np.float32([pt[0] for pt in correspondences]).reshape(-1, 1, 2)
            tracked_pts_r = np.float32([pt[1] for pt in correspondences]).reshape(-1, 1, 2)

        else:
            tracked_pts_l, _ = Track_Keypoints(prev_gray_l, left_img, tracked_pts_l)
            tracked_pts_r, _ = Track_Keypoints(prev_gray_r, right_img, tracked_pts_r)

        prev_gray_l, prev_gray_r = left_img.copy(), right_img.copy()

        # Triangulate
        if tracked_pts_l.shape[0] > 0 and tracked_pts_l.shape == tracked_pts_r.shape:
            pts_l = tracked_pts_l.reshape(-1, 2)
            pts_r = tracked_pts_r.reshape(-1, 2)
            correspondences = list(zip(pts_l, pts_r))
            pcd = Triangulate_Points(correspondences, Q)

            for point in pcd:
                x, y, z = point
                if z > 0 and z < 3000:
                    cv2.circle(frame, (int(x / z * 100 + 640), int(y / z * 100)), 2, (0, 255, 0), -1)

        cv2.imshow("Stereo Track + PCD", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# ===== Main =====
if __name__ == "__main__":
    Run_Stereo_Tracking()