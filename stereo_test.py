import cv2
import numpy as np
import json
import apriltag
import open3d as o3d
from utility import stereo_utils as su

# ===== Global Variables =====
JSON_PATH = "camera_calibration_results.json"
VIDEO_SOURCE = 0  # Change if needed
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480

# Load camera calibration data
def load_calibration():
    with open(JSON_PATH, "r") as f:
        calib_data = json.load(f)

    mtx_l = np.array(calib_data["mtx_left"])
    dist_l = np.array(calib_data["dist_left"])
    mtx_r = np.array(calib_data["mtx_right"])
    dist_r = np.array(calib_data["dist_right"])
    R = np.array(calib_data["R"])
    T = np.array(calib_data["T"])
    Q = np.array(calib_data["Q"])
    
    return mtx_l, dist_l, mtx_r, dist_r, R, T, Q

# Detect AprilTags in both images
def detect_apriltags(left_img, right_img):
    detector = apriltag.Detector()
    
    detections_left = detector.detect(left_img)
    detections_right = detector.detect(right_img)

    correspondences = []
    for left_tag in detections_left:
        for right_tag in detections_right:
            if left_tag.tag_id == right_tag.tag_id:
                correspondences.append((left_tag.corners, right_tag.corners))
    
    return correspondences

# Triangulate points from stereo correspondences
def triangulate_points(correspondences, mtx_l, mtx_r, dist_l, dist_r, R, T, Q):
    points_3d = []

    # Get projection matrices from calibration data
    P1 = np.dot(mtx_l, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(mtx_r, np.hstack((R, T)))

    for left_corners, right_corners in correspondences:
        # Convert to homogeneous coordinates
        left_homogeneous = np.array([left_corners[0][0], left_corners[0][1], 1])
        right_homogeneous = np.array([right_corners[0][0], right_corners[0][1], 1])

        # Triangulate the points
        point_3d_homogeneous = cv2.triangulatePoints(P1, P2, left_homogeneous, right_homogeneous)

        # Convert from homogeneous to 3D coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
        points_3d.append(point_3d)

    return points_3d

# Create and visualize point cloud
def visualize_point_cloud(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    o3d.visualization.draw_geometries([pcd])

# Main function
def main():
    # Load calibration data
    mtx_l, dist_l, mtx_r, dist_r, R, T, Q = load_calibration()

    # Open camera feed
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        # Split stereo frame into left and right images
        left_img = frame[:, :640]
        right_img = frame[:, 640:]

        # Convert to grayscale
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in both images
        correspondences = detect_apriltags(gray_left, gray_right)
        
        if len(correspondences) > 0:
            # Triangulate points and visualize
            points_3d = triangulate_points(correspondences, mtx_l, mtx_r, dist_l, dist_r, R, T, Q)
            visualize_point_cloud(points_3d)
        
        # Show the stereo images
        cv2.imshow("Left Image", left_img)
        cv2.imshow("Right Image", right_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
