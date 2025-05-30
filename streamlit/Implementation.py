import os
import av
import cv2
import torch
import numpy as np
import streamlit as st
import tempfile
import open3d as o3d
from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utility import inference_model

# ====== CONFIG ======
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "baked")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 244

# ====== IMAGE TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class StereoVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Assuming 1280x480 stereo frame
        try:
            pred_frame, render_frame = predict_and_render(model, img)
            self.result_frame = render_frame
            return av.VideoFrame.from_ndarray(pred_frame, format="bgr24")
        except Exception as e:
            print(f"[WebRTC Error] {e}")
            return frame
        
# ====== 3D VISUALIZATION ======
def render_3d_pose(pose):
    def create_colored_face(width, height, color, transform):
        mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=0.001)
        mesh.paint_uniform_color(color)
        mesh.translate(transform)
        return mesh

    # Define face colors
    face_colors = {
        "front":  [1, 0, 0],    # Red
        "back":   [0, 1, 0],    # Green
        "left":   [0, 0, 1],    # Blue
        "right":  [1, 1, 0],    # Yellow
        "top":    [1, 0, 1],    # Magenta
        "bottom": [0, 1, 1],    # Cyan
    }

    # Define face meshes (each face is a plane)
    faces = [
        create_colored_face(0.1, 0.1, face_colors["front"],  [-0.05, -0.05,  0.05]),
        create_colored_face(0.1, 0.1, face_colors["back"],   [-0.05, -0.05, -0.05]),
        create_colored_face(0.001, 0.1, face_colors["left"], [-0.05, -0.05, -0.05]),
        create_colored_face(0.001, 0.1, face_colors["right"],[ 0.05, -0.05, -0.05]),
        create_colored_face(0.1, 0.001, face_colors["bottom"],[-0.05, -0.05, -0.05]),
        create_colored_face(0.1, 0.001, face_colors["top"],   [-0.05,  0.05, -0.05])
    ]

    # Merge all face meshes into one cube
    cube = faces[0]
    for face in faces[1:]:
        cube += face

    # Apply pose
    trans = pose[:3]
    rot = np.radians(pose[3:6])  # Convert degrees to radians
    R = o3d.geometry.get_rotation_matrix_from_xyz(rot)
    cube.rotate(R, center=(0, 0, 0))
    cube.translate(trans)

    # Visualize to image
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=400, height=400)
    vis.add_geometry(cube)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    return (np.asarray(img) * 255).astype(np.uint8)

# ====== MODEL LOADING ======
def get_available_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]

# ====== STREAMLIT UI ======
st.title("6D Pose Estimation - Implementation Demo")

mode = st.radio("Input Source", ["Stereo Webcam", "Upload Video", "Use Local Video", "Upload Image"])
model_file = st.selectbox("Select Model", get_available_models())
model = inference_model.get_model(os.path.join(MODEL_DIR, model_file))

video_frame = st.empty()
overlay_frame = st.empty()

# ====== VIDEO FRAME TO POSE PIPELINE ======
def predict_and_render(model, frame):
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

# ====== VIDEO HANDLING ======
cap = None
if mode == "Upload Video":
    uploaded = st.file_uploader("Upload a stereo video", type=["mp4", "avi"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)

elif mode == "Use Local Video":
    raw_video_dir = os.path.join(os.path.dirname(__file__), "..", "video", "raw")
    available_videos = [f for f in os.listdir(raw_video_dir) if f.endswith((".mp4", ".avi"))]

    if available_videos:
        selected_video = st.selectbox("Select a video from raw/", available_videos)
        selected_path = os.path.join(raw_video_dir, selected_video)
        cap = cv2.VideoCapture(selected_path)
    else:
        st.warning("⚠️ No videos found in video/raw/")


elif mode == "Stereo Webcam":
    st.warning("⚠️ Using client webcam via WebRTC.")
    ctx = webrtc_streamer(
        key="stereo-webrtc",
        video_processor_factory=StereoVideoProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 480}},
        async_processing=True,
    )

    if ctx.video_processor and ctx.video_processor.result_frame is not None:
        overlay_frame.image(ctx.video_processor.result_frame, caption="3D Pose", channels="RGB")

elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload a stereo image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)
        pred_frame, render_frame = predict_and_render(model, frame)
        video_frame.image(pred_frame, caption="Input Frame", channels="RGB")
        overlay_frame.image(render_frame, caption="3D Pose", channels="RGB")

# ====== PROCESS LOOP ======
if cap is not None and cap.isOpened():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pred_frame, render_frame = predict_and_render(model, frame)
        video_frame.image(pred_frame, channels="RGB")
        overlay_frame.image(render_frame, channels="RGB")

    cap.release()
else:
    if mode != "Upload Image":
        st.warning("⏳ Waiting for video input...")
