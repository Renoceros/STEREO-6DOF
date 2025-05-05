#stereo_utils.py
from datetime import datetime
import json
import os
import winsound
import cv2
import numpy as np

# ===== Operation Critical functions =====

def OpenCam(video_source=0,fw=1280,fh=480):
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh) 
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {orig_width}x{orig_height}")
    Ding()
    return cap, orig_width, orig_height

def LoJ(key,json_path):
    """Load a variable from the JSON state file."""
    if not os.path.exists(json_path):
        print("[LoJ] State file does not exist.")
        return None
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get(key)

def UpJ(key, value,json_path):
    """Update a variable in the JSON state file."""
    data = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    data[key] = value
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[UpJ] {key} updated to {value}")

def Split_Stereo_Frame(frame):
    height, width = frame.shape[:2]
    mid = width // 2
    left = frame[:, :mid]
    right = frame[:, mid:]
    return left, right

def load_calibration(JSON_PATH = "camera_calibration_results.json"):
    with open(JSON_PATH, 'r') as f:
        calib_data = json.load(f)
    return calib_data

# ===== Nice to haves functions =====

def Ding():
    winsound.MessageBeep(winsound.MB_ICONASTERISK)
    return

def Deng():
    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    return

def Current():
    current = datetime.now()
    current_str = datetime.now().strftime("%H:%M:%S")
    return current, current_str

def Convert_to_Grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def Edge(image):
    #GRAY
    if len(image.shape) == 3 and image.shape[2] == 3:
            # Video is color, convert to gray first
            image_gay = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        image_gay = image.copy()
    #image_gay = Convert_to_Grayscale(image)
    # Compute the gradient of the image using Sobel operator
    sobel_x = cv2.Sobel(image_gay, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(image_gay, cv2.CV_64F, 0, 1, ksize=1)
    # Calculate the magnitude of the gradient (the contrast)
    g_m = cv2.magnitude(sobel_x, sobel_y)
    # Normalize the gradient to range 0-255
    g_m_n = cv2.normalize(g_m, None, 0, 255, cv2.NORM_MINMAX)
    # Convert back to uint8 for visualization purposes ofc ofc
    g_m_n = np.uint8(g_m_n)
    # Bump contrast by scaling pixel values
    alpha = 3  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)
    # Apply contrast and brightness adjustment
    g_m_n_c = cv2.convertScaleAbs(g_m_n, alpha=alpha, beta=beta)
    # Blend using a weighted addition (tweak alpha as needed)
    # x = 0.5
    # blended = cv2.addWeighted(image_gay, x, g_m_n_c, 1 - x, 0)
    return g_m_n_c