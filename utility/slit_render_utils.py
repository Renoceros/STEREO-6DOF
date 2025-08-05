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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    model_name = model_file.lower()

    if "6ch" in model_name:
        mid = w // 2
        left = transform(Image.fromarray(frame[:, :mid]))
        right = transform(Image.fromarray(frame[:, mid:]))
        stacked = torch.cat([left, right], dim=0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(stacked).cpu().numpy()[0]
        combined = cv2.hconcat([frame[:, :mid], frame[:, mid:]])
    elif "sw" in model_name:
        mid = w // 2
        left = frame[:, :mid]
        right = frame[:, mid:]
        left_img = transform(Image.fromarray(left)).unsqueeze(0).to(DEVICE)
        right_img = transform(Image.fromarray(right)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(left_img, right_img).cpu().numpy()[0]
        combined = cv2.hconcat([left, right])
    else:
        img_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor).cpu().numpy()[0]
        combined = frame

    render = render_3d_pose(output)
    return combined, render

def to_av_frame(pred_frame):
    return av.VideoFrame.from_ndarray(cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR), format="bgr24")

def render_3d_pose(pose):
    def create_colored_face(width, height, color, transform):
        mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=0.001)
        mesh.paint_uniform_color(color)
        mesh.translate(transform)
        return mesh

    face_colors = {
        "front":  [1, 0, 0],
        "back":   [0, 1, 0],
        "left":   [0, 0, 1],
        "right":  [1, 1, 0],
        "top":    [1, 0, 1],
        "bottom": [0, 1, 1],
    }

    faces = [
        create_colored_face(0.1, 0.1, face_colors["front"],  [-0.05, -0.05,  0.05]),
        create_colored_face(0.1, 0.1, face_colors["back"],   [-0.05, -0.05, -0.05]),
        create_colored_face(0.001, 0.1, face_colors["left"], [-0.05, -0.05, -0.05]),
        create_colored_face(0.001, 0.1, face_colors["right"],[ 0.05, -0.05, -0.05]),
        create_colored_face(0.1, 0.001, face_colors["bottom"],[-0.05, -0.05, -0.05]),
        create_colored_face(0.1, 0.001, face_colors["top"],   [-0.05,  0.05, -0.05])
    ]

    cube = faces[0]
    for face in faces[1:]:
        cube += face

    trans = pose[:3]
    rot = np.radians(pose[3:6])
    R = o3d.geometry.get_rotation_matrix_from_xyz(rot)
    cube.rotate(R, center=(0, 0, 0))
    cube.translate(trans)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=400, height=400)
    vis.add_geometry(cube)

    vis.poll_events()
    vis.update_renderer()

    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    view_ctl.rotate(10.0, 0.0)

    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    rgb_img = (np.asarray(img) * 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return bgr_img
