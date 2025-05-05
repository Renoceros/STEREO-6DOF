# aruco_box_relative_pose_estimation.py
import cv2
import json
import numpy as np
import os
import utility.stereo_utils as su
from collections import defaultdict
from aruco_pose_estimation import FACE_NAMES, estimate_marker_pose

# Cube dimensions
CUBE_SIZE = 0.08  # 8cm cube

# === CONFIG ===
CAM_ID = "video/raw/ArUco.avi"
CAM_WIDTH = 1280
CAM_HEIGHT = 480
MARKER_LENGTH_M = 0.0715  # 7.15cm marker width
CALIB_JSON = "development_calibration.json"
ARUCO_DIR = "ArUco/"

# Load face axes configuration
try:
    with open("face_axes.json") as f:
        FACE_AXES = json.load(f)
except:
    FACE_AXES = {
        "front": [[1,0,0], [0,1,0], [0,0,-1]],
        "back": [[-1,0,0], [0,1,0], [0,0,1]],
        "left": [[0,0,-1], [0,1,0], [-1,0,0]],
        "right": [[0,0,1], [0,1,0], [1,0,0]],
        "top": [[-1,0,0], [0,0,1], [0,-1,0]],
        "bottom": [[-1,0,0], [0,0,-1], [0,1,0]]
    }

# === LOAD CAMERA CALIBRATION ===
with open(CALIB_JSON, 'r') as f:
    calib = json.load(f)
mtxL = np.array(calib['mtx_left'])
distL = np.array(calib['dist_left'])
mtxR = np.array(calib['mtx_right'])
distR = np.array(calib['dist_right'])
T = np.array(calib['T'])  # Stereo baseline translation

# === SETUP ARUCO ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector = cv2.aruco.ArucoDetector(aruco_dict)

def get_cube_center_pose(marker_poses):
    """Calculate cube center pose from multiple marker poses"""
    if not marker_poses:
        return None, None
    
    # Convert all poses to the same coordinate system
    rotations = []
    translations = []
    
    for face_name, (rvec, tvec) in marker_poses.items():
        # Get face offset from cube center (assuming markers are centered on faces)
        face_offset = np.zeros(3)
        if face_name == "front":
            face_offset = [0, 0, CUBE_SIZE/2]
        elif face_name == "back":
            face_offset = [0, 0, -CUBE_SIZE/2]
        elif face_name == "left":
            face_offset = [-CUBE_SIZE/2, 0, 0]
        elif face_name == "right":
            face_offset = [CUBE_SIZE/2, 0, 0]
        elif face_name == "top":
            face_offset = [0, CUBE_SIZE/2, 0]
        elif face_name == "bottom":
            face_offset = [0, -CUBE_SIZE/2, 0]
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate cube center position
        cube_center = tvec - R @ face_offset.reshape(3,1)
        translations.append(cube_center)
        rotations.append(R)
    
    # Average all estimates (simple approach - could use more sophisticated fusion)
    avg_translation = np.mean(translations, axis=0)
    avg_rotation = np.mean(rotations, axis=0)
    
    # Convert back to rotation vector
    avg_rvec, _ = cv2.Rodrigues(avg_rotation)
    
    return avg_rvec, avg_translation

def process_frame(frame):
    """Process frame and return cube pose"""
    left, right = su.Split_Stereo_Frame(frame)
    left_undist = cv2.undistort(left, mtxL, distL)
    right_undist = cv2.undistort(right, mtxR, distR)
    
    # Detect markers in both views
    grayL = cv2.cvtColor(left_undist, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_undist, cv2.COLOR_BGR2GRAY)
    
    cornersL, idsL, _ = detector.detectMarkers(grayL)
    cornersR, idsR, _ = detector.detectMarkers(grayR)
    
    marker_poses = {}
    
    # Process left view markers
    if idsL is not None:
        for i, marker_id in enumerate(idsL.flatten()):
            face_name = FACE_NAMES.get(int(marker_id))
            if face_name:
                rvec, tvec = estimate_marker_pose(cornersL[i],MARKER_LENGTH_M, mtxL, distL)
                if rvec is not None:
                    marker_poses[f"left_{face_name}"] = (rvec, tvec)
    
    # Process right view markers
    if idsR is not None:
        for i, marker_id in enumerate(idsR.flatten()):
            face_name = FACE_NAMES.get(int(marker_id))
            if face_name:
                rvec, tvec = estimate_marker_pose(cornersR[i],MARKER_LENGTH_M, mtxR, distR)
                if rvec is not None:
                    marker_poses[f"right_{face_name}"] = (rvec, tvec)
    
    # Calculate cube center pose
    cube_rvec, cube_tvec = get_cube_center_pose(marker_poses)
    
    return cube_rvec, cube_tvec, marker_poses

def transform_to_global_pose(tvec_left, tvec_right):
    """Convert left/right camera poses to global stereo coordinate system"""
    # Midpoint between cameras is origin (assuming cameras are aligned on X-axis)
    pose_global = (tvec_left + tvec_right) / 2
    pose_global[0] += T[0]/2  # Adjust for baseline
    
    return pose_global

if __name__ == "__main__":
    cap, _, _ = su.OpenCam(CAM_ID)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cube_rvec, cube_tvec, _ = process_frame(frame)
            
            if cube_rvec is not None:
                print(f"Cube position: {cube_tvec.flatten()}") 
                print(f"Cube rotation: {np.degrees(cube_rvec.flatten())}°")
                
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()