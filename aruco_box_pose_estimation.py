# aruco_box_pose_estimation.py
import cv2
import json
import numpy as np
from utility.stereo_utils import Split_Stereo_Frame, OpenCam
from aruco_pose_estimation import FACE_NAMES, estimate_marker_pose

# Cube dimensions
CUBE_SIZE = 0.08  # 8cm cube

class CubePoseEstimator:
    def __init__(self, config_path="development_calibration.json"):
        # Load calibration data
        with open(config_path) as f:
            calib = json.load(f)
        
        self.mtxL = np.array(calib['mtx_left'])
        self.distL = np.array(calib['dist_left'])
        self.mtxR = np.array(calib['mtx_right'])
        self.distR = np.array(calib['dist_right'])
        self.T = np.array(calib['T'])  # Stereo baseline
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        
        # Load face axes configuration
        try:
            with open("face_axes.json") as f:
                self.FACE_AXES = json.load(f)
        except:
            self.FACE_AXES = {
                "front": [[1,0,0], [0,1,0], [0,0,-1]],
                "back": [[-1,0,0], [0,1,0], [0,0,1]],
                "left": [[0,0,-1], [0,1,0], [-1,0,0]],
                "right": [[0,0,1], [0,1,0], [1,0,0]],
                "top": [[-1,0,0], [0,0,1], [0,-1,0]],
                "bottom": [[-1,0,0], [0,0,-1], [0,1,0]]
            }

    def get_cube_center_pose(self, marker_poses):
        """Calculate cube center pose from multiple marker poses"""
        if not marker_poses:
            return None, None
        
        rotations = []
        translations = []
        
        for face_name, (rvec, tvec) in marker_poses.items():
            # Face offset from cube center
            face_offset = self._get_face_offset(face_name)
            
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Calculate cube center position
            cube_center = tvec - R @ face_offset.reshape(3,1)
            translations.append(cube_center)
            rotations.append(R)
        
        # Average all estimates
        avg_translation = np.mean(translations, axis=0)
        avg_rotation = np.mean(rotations, axis=0)
        
        # Convert back to rotation vector
        avg_rvec, _ = cv2.Rodrigues(avg_rotation)
        
        return avg_rvec, avg_translation

    def _get_face_offset(self, face_name):
        """Get the 3D offset vector for each face"""
        offset = np.zeros(3)
        if face_name == "front":
            offset = [0, 0, CUBE_SIZE/2]
        elif face_name == "back":
            offset = [0, 0, -CUBE_SIZE/2]
        elif face_name == "left":
            offset = [-CUBE_SIZE/2, 0, 0]
        elif face_name == "right":
            offset = [CUBE_SIZE/2, 0, 0]
        elif face_name == "top":
            offset = [0, CUBE_SIZE/2, 0]
        elif face_name == "bottom":
            offset = [0, -CUBE_SIZE/2, 0]
        return np.array(offset)

    def process_frame(self, frame):
        """Process frame and return cube pose"""
        left, right = Split_Stereo_Frame(frame)
        left_undist = cv2.undistort(left, self.mtxL, self.distL)
        right_undist = cv2.undistort(right, self.mtxR, self.distR)
        
        # Detect markers in both views
        grayL = cv2.cvtColor(left_undist, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_undist, cv2.COLOR_BGR2GRAY)
        
        cornersL, idsL, _ = self.detector.detectMarkers(grayL)
        cornersR, idsR, _ = self.detector.detectMarkers(grayR)
        
        marker_poses = {}
        
        # Process left view markers
        if idsL is not None:
            for i, marker_id in enumerate(idsL.flatten()):
                face_name = FACE_NAMES.get(int(marker_id))
                if face_name:
                    rvec, tvec = estimate_marker_pose(cornersL[i], 0.0715, self.mtxL, self.distL)
                    if rvec is not None:
                        marker_poses[face_name] = (rvec, tvec)
        
        # Process right view markers
        if idsR is not None:
            for i, marker_id in enumerate(idsR.flatten()):
                face_name = FACE_NAMES.get(int(marker_id))
                if face_name:
                    rvec, tvec = estimate_marker_pose(cornersR[i], 0.0715, self.mtxR, self.distR)
                    if rvec is not None:
                        marker_poses[face_name] = (rvec, tvec)
        
        # Calculate cube center pose
        return self.get_cube_center_pose(marker_poses)

    def transform_to_global_pose(self, tvec_left, tvec_right):
        """Convert to global stereo coordinate system"""
        pose_global = (tvec_left + tvec_right) / 2
        pose_global[0] += self.T[0]/2  # Adjust for baseline
        return pose_global

if __name__ == "__main__":
    pose_estimator = CubePoseEstimator()
    cap, _, _ = OpenCam("video/raw/ArUco.avi")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cube_rvec, cube_tvec = pose_estimator.process_frame(frame)
            
            if cube_rvec is not None:
                print(f"Cube position: {cube_tvec.flatten()}") 
                print(f"Cube rotation: {np.degrees(cube_rvec.flatten())}°")
                
            cv2.imshow("Cube Pose Estimation", frame)
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()