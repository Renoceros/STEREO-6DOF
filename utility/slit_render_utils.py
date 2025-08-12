# utility/slit_render_utils.py
import os
import cv2
import av
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import open3d as o3d
from streamlit_webrtc import VideoProcessorBase
from utility import inference_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 244

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_available_models(model_dir):
    return [f for f in os.listdir(model_dir) if f.endswith(".pt")]

def load_model(model_dir, model_file):
    return inference_model.get_model(os.path.join(model_dir, model_file))

def predict_and_render(model, model_file, frame):
    # The input 'frame' is expected to be in BGR format from OpenCV.
    
    # Convert to RGB for PIL and model processing.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    model_name = model_file.lower()

    if "6ch" in model_name:
        mid = w // 2
        left = transform(Image.fromarray(frame_rgb[:, :mid]))
        right = transform(Image.fromarray(frame_rgb[:, mid:]))
        stacked = torch.cat([left, right], dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(stacked).cpu().numpy()[0]
        # The 'combined' frame is currently RGB.
        combined_rgb = cv2.hconcat([frame_rgb[:, :mid], frame_rgb[:, mid:]])
        
    elif "sw" in model_name:
        mid = w // 2
        left = frame_rgb[:, :mid]
        right = frame_rgb[:, mid:]
        left_img = transform(Image.fromarray(left)).unsqueeze(0).to(DEVICE)
        right_img = transform(Image.fromarray(right)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(left_img, right_img).cpu().numpy()[0]
        # The 'combined' frame is currently RGB.
        combined_rgb = cv2.hconcat([left, right])
        
    else:
        img_tensor = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor).cpu().numpy()[0]
        # The 'combined' frame is currently RGB.
        combined_rgb = frame_rgb

    # Get the 3D render, which is already in BGR format.
    render_bgr = render_3d_pose(output)
    
    # FIX: Convert the combined frame from RGB back to BGR before returning.
    # This ensures both returned frames are in the same BGR format.
    combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
    
    return combined_bgr, render_bgr

def to_av_frame(bgr_frame):
    # FIX: This function now expects a BGR frame, so no conversion is needed.
    # It passes the frame directly to the video streamer.
    return av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")

def render_3d_pose(pose_9d):
    # --- 1. EXTRACT POSE & CONVERT 6D ROTATION TO 3x3 MATRIX ---
    trans = pose_9d[:3]
    rot_6d = pose_9d[3:]
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack((b1, b2, b3), axis=1)

    # --- 2. LOAD AND PREPARE MESH ---
    mesh = o3d.io.read_triangle_mesh("cube.glb", enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    mesh.scale(0.04, center=mesh.get_center())
    correction_rotation = o3d.geometry.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
    mesh.rotate(correction_rotation, center=mesh.get_center())

    # --- 3. APPLY THE PREDICTED POSE ---
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate(trans)

    # --- 4. VISUALIZATION ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=800)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1])
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic.copy()
    extrinsic[:3, 3] = np.array([0.06, 0.0, 0.8])
    cam_params.extrinsic = extrinsic
    view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    # Convert the rendered RGB image to BGR before returning.
    rgb_img = (np.asarray(img) * 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # This function correctly returns a BGR image.
    return bgr_img
