# aruco_pose_estimation.py
import cv2
import json
import numpy as np
import os
import utility.stereo_utils as su

# === CONFIG ===
CAM_ID = "video/raw/ArUco.mp4"
CAM_WIDTH = 1280
CAM_HEIGHT = 480
MARKER_LENGTH_M = 0.0715  # 7.15cm marker width
CALIB_JSON = "development_calibration.json"
ARUCO_DIR = "ArUco/"

# === FACE ID MAPPING ===
FACE_NAMES = {
    0: "front",
    1: "back",
    2: "left",
    3: "right",
    4: "top",
    5: "bottom"
}

# Face-specific axis definitions (X, Y, Z vectors)
FACE_AXES = {
    "front": {
        'x': [1, 0, 0],   # Right
        'y': [0, 1, 0],   # Up
        'z': [0, 0, -1]   # Out (toward viewer)
    },
    "back": {
        'x': [-1, 0, 0],  # Left
        'y': [0, 1, 0],   # Up
        'z': [0, 0, 1]    # Out (away from viewer)
    },
    "left": {
        'x': [0, 0, -1],   # Toward back
        'y': [0, 1, 0],   # Up
        'z': [-1, 0, 0]   # Out (left side)
    },
    "right": {
        'x': [0, 0, 1],  # Toward front
        'y': [0, 1, 0],   # Up
        'z': [1, 0, 0]    # Out (right side)
    },
    "top": {
        'x': [-1, 0, 0],   # Right
        'y': [0, 0, 1],  # Toward front
        'z': [0, -1, 0]    # Up
    },
    "bottom": {
        'x': [-1, 0, 0],   # Right
        'y': [0, 0, -1],   # Toward back
        'z': [0, 1, 0]   # Down
    }
}
# === LOAD CAMERA CALIBRATION ===
try:
    with open(CALIB_JSON, 'r') as f:
        calib = json.load(f)
    mtxL = np.array(calib['mtx_left'])
    distL = np.array(calib['dist_left'])
    mtxR = np.array(calib['mtx_right'])
    distR = np.array(calib['dist_right'])
except Exception as e:
    su.Deng()
    raise RuntimeError(f"Failed to load calibration: {e}")

# === SETUP ARUCO ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector = cv2.aruco.ArucoDetector(aruco_dict)

def Build_Face_ID_Map():
    """Improved marker validation with error handling"""
    face_id_map = {}
    for fname in os.listdir(ARUCO_DIR):
        if not fname.lower().endswith(".png"):
            continue
            
        face_name = os.path.splitext(fname)[0]
        img_path = os.path.join(ARUCO_DIR, fname)
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Couldn't read {img_path}")
                
            corners, ids, _ = detector.detectMarkers(img)
            
            if ids is None or len(ids) != 1:
                raise ValueError(f"Marker detection failed in {fname}")
                
            marker_id = int(ids[0][0])
            if marker_id not in FACE_NAMES:
                raise ValueError(f"Unknown marker ID {marker_id} in {fname}")
                
            face_id_map[marker_id] = face_name
            print(f"✅ Mapped {face_name} to ID {marker_id}")
            
        except Exception as e:
            su.Deng()
            print(f"⚠️ Error processing {fname}: {str(e)}")
            
    return face_id_map

def Process_Frame(frame, face_id_map):
    """Process stereo frame with error handling"""
    try:
        # Split and undistort
        left_raw, right_raw = su.Split_Stereo_Frame(frame)
        left_undist = cv2.undistort(left_raw, mtxL, distL)
        right_undist = cv2.undistort(right_raw, mtxR, distR)
        
        # Process both views
        left_det = process_single_view(left_undist, mtxL, distL, face_id_map, "L")
        right_det = process_single_view(right_undist, mtxR, distR, face_id_map, "R")
        
        return np.hstack((left_det, right_det))
    except Exception as e:
        
        print(f"Frame processing error: {e}")
        return frame

def process_single_view(img, mtx, dist, face_id_map, side_prefix=""):
    """Process a single camera view with corrected pose estimation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    
    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        return img
    
    # Draw markers and estimate pose for each
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    
    for i, marker_id in enumerate(ids.flatten()):
        face_name = face_id_map.get(int(marker_id), f"ID:{marker_id}")
        
        # Estimate pose
        rvec, tvec = estimate_marker_pose(corners[i], MARKER_LENGTH_M, mtx, dist)
        
        if rvec is not None:
            # Draw axes with face-specific orientation
            draw_face_axes(img, mtx, dist, rvec, tvec, face_name)
            
            # Label with face name
            text_pos = tuple(corners[i][0][0].astype(int))
            cv2.putText(
                img, f"{side_prefix}:{face_name}",
                text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,255), 2
            )
    
    return img

def estimate_marker_pose(corners, marker_length, camera_matrix, dist_coeffs):
    """Estimate marker pose using solvePnP"""
    # Define marker corners in 3D space (Z=0 plane)
    obj_points = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)
    
    # Reshape detected corners
    img_points = corners.reshape(-1, 2)
    
    # Get pose estimate
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if not success:
        return None, None
    
    return rvec, tvec

def draw_face_axes(img, mtx, dist, rvec, tvec, face_name):
    """Draw axes with distinct colors for positive/negative directions"""
    axis_length = MARKER_LENGTH_M * 0.7
    
    # Get face-specific axis directions
    if face_name not in FACE_AXES:
        face_name = "front"  # Default to front if unknown
    
    axes_def = FACE_AXES[face_name]
    
    # Create both positive and negative axes
    axes = np.float32([
        [0, 0, 0],         # Origin point
        axes_def['x'],     # +X
        [-x for x in axes_def['x']],  # -X
        axes_def['y'],     # +Y
        [-y for y in axes_def['y']],  # -Y
        axes_def['z'],     # +Z
        [-z for z in axes_def['z']]   # -Z
    ]) * axis_length
    
    # Project 3D points to 2D
    imgpts, _ = cv2.projectPoints(axes, rvec, tvec, mtx, dist)
    origin = tuple(imgpts[0].ravel().astype(int))
    
    # Color scheme:
    # X: Red (+) / Pink (-)
    # Y: Green (+) / Light Green (-)
    # Z: Blue (+) / Cyan (-)
    cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)   # +X (red)
    cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (180, 105, 255), 3) # -X (pink)
    cv2.line(img, origin, tuple(imgpts[3].ravel().astype(int)), (0, 255, 0), 3)   # +Y (green)
    cv2.line(img, origin, tuple(imgpts[4].ravel().astype(int)), (144, 238, 144), 3) # -Y (light green)
    cv2.line(img, origin, tuple(imgpts[5].ravel().astype(int)), (255, 0, 0), 3)   # +Z (blue)
    cv2.line(img, origin, tuple(imgpts[6].ravel().astype(int)), (255, 255, 0), 3)  # -Z (cyan)
    
    # Label the positive axes
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '+X', tuple(imgpts[1].ravel().astype(int)), font, 0.4, (0,0,255), 1)
    cv2.putText(img, '+Y', tuple(imgpts[3].ravel().astype(int)), font, 0.4, (0,255,0), 1)
    cv2.putText(img, '+Z', tuple(imgpts[5].ravel().astype(int)), font, 0.4, (255,0,0), 1)

if __name__ == "__main__":
    # su.Ding()
    print("=== Starting ArUco Pose Estimation ===")
    
    face_id_map = Build_Face_ID_Map()
    if not face_id_map:
        # su.Deng()
        raise RuntimeError("No valid markers found in ArUco directory")
    
    cap, _, _ = su.OpenCam(CAM_ID)
    if not cap.isOpened():
        # su.Deng()
        raise RuntimeError("Failed to open camera")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # su.Deng()
                print("Camera feed ended")
                break
                
            processed = Process_Frame(frame, face_id_map)
            cv2.imshow("Stereo ArUco Pose Estimation", processed)
            
            if cv2.waitKey(1) in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("=== Clean exit ===")