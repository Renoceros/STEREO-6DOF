# Implementation.py
import os
import cv2
import numpy as np
import streamlit as st
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import time
import threading

import torch


# It's assumed that 'slit_render_utils' is in a directory named 'utility'
# and accessible from this script's location.
from utility import slit_render_utils as utils

# ====== CONFIGURATION ======
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "baked")

# ====== STREAMLIT UI SETUP ======
st.set_page_config(layout="wide")
st.title("6D Pose Estimation - Implementation Demo")

# --- Main Page Controls ---
st.header("Controls")
col1_controls, col2_controls = st.columns(2)

with col1_controls:
    mode = st.radio(
        "Input Source",
        ["Stereo Webcam", "Upload Video", "Use Local Video", "Upload Image"]
    )

with col2_controls:
    model_file = st.selectbox(
        "Select Model",
        utils.get_available_models(MODEL_DIR)
    )

# Load the selected PyTorch model
@st.cache_resource
def load_model(model_path):
    return utils.load_model(MODEL_DIR, model_path)

model = load_model(model_file)

# --- Main Display Area ---
col1_display, col2_display = st.columns(2)
with col1_display:
    video_frame_placeholder = st.empty()
with col2_display:
    render_frame_placeholder = st.empty()

# ====== WEBRTC VIDEO PROCESSOR (MODIFIED) ======
class StereoVideoProcessor(utils.VideoProcessorBase):
    def __init__(self, model, model_file):
        self.model = model
        self.model_file = model_file
        self.latest_pose = None
        # A lock is good practice for thread-safe access to latest_pose
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # This function now only does the prediction, not the rendering.
        # It's lighter and safer to run in the background thread.
        pred_frame, pose_9d = utils.predict_pose_only(self.model, self.model_file, img)
        
        # Store the latest pose in a thread-safe way.
        with self.lock:
            self.latest_pose = pose_9d
            
        # Only return the webcam frame to the video player.
        return utils.to_av_frame(pred_frame)

# We need a modified utility function that only returns the pose
# Let's add it to the utils file conceptually, or define it here if needed.
# For simplicity, let's assume `predict_pose_only` exists in utils.
# It would be a copy of `predict_and_render` that stops before calling `render_3d_pose`.

# Let's define a local version of `predict_pose_only` inside slit_render_utils.py
# You should add this function to your actual utility file.
def predict_pose_only(model, model_file, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    model_name = model_file.lower()
    # ... (same model inference logic as before) ...
    if "6ch" in model_name:
        mid = w // 2
        left = utils.transform(Image.fromarray(frame_rgb[:, :mid]))
        right = utils.transform(Image.fromarray(frame_rgb[:, mid:]))
        stacked = torch.cat([left, right], dim=0).unsqueeze(0).to(utils.DEVICE)
        with torch.no_grad():
            output = model(stacked).cpu().numpy()[0]
        combined_rgb = cv2.hconcat([frame_rgb[:, :mid], frame_rgb[:, mid:]])
    elif "sw" in model_name:
        mid = w // 2
        left = frame_rgb[:, :mid]
        right = frame_rgb[:, mid:]
        left_img = utils.transform(Image.fromarray(left)).unsqueeze(0).to(utils.DEVICE)
        right_img = utils.transform(Image.fromarray(right)).unsqueeze(0).to(utils.DEVICE)
        with torch.no_grad():
            output = model(left_img, right_img).cpu().numpy()[0]
        combined_rgb = cv2.hconcat([left, right])
    else:
        img_tensor = utils.transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(utils.DEVICE)
        with torch.no_grad():
            output = model(img_tensor).cpu().numpy()[0]
        combined_rgb = frame_rgb
        
    combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
    return combined_bgr, output

# Monkey-patch the function into the utils module for this script run
utils.predict_pose_only = predict_pose_only


# ====== MAIN LOGIC FOR INPUT MODES ======
cap = None

if mode == "Stereo Webcam":
    ctx = webrtc_streamer(
        key="stereo-webrtc",
        video_processor_factory=lambda: StereoVideoProcessor(model=model, model_file=model_file),
        media_stream_constraints={"video": {"width": 1280, "height": 480}},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}),
    )
    
    # This loop runs in the MAIN THREAD.
    while ctx.state.playing:
        pose_to_render = None
        # Safely get the latest pose from the background thread.
        if ctx.video_processor:
            with ctx.video_processor.lock:
                pose_to_render = ctx.video_processor.latest_pose
        
        # If there is a pose, render it in the main thread.
        if pose_to_render is not None:
            render_frame = utils.render_3d_pose(pose_to_render)
            render_frame_placeholder.image(render_frame, caption="3D Pose", channels="BGR")
        
        # Sleep to prevent this loop from running too fast and using 100% CPU.
        time.sleep(0.01)


elif mode == "Upload Video" or mode == "Use Local Video":
    if mode == "Upload Video":
        uploaded = st.file_uploader("Upload a stereo video", type=["mp4", "avi", "mov"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
    else: # Use Local Video
        raw_video_dir = os.path.join(os.path.dirname(__file__), "..", "video", "raw")
        available_videos = [f for f in os.listdir(raw_video_dir) if f.endswith((".mp4", ".avi", ".mov"))]
        if available_videos:
            selected_video = st.selectbox("Select a video from raw/", available_videos)
            selected_path = os.path.join(raw_video_dir, selected_video)
            cap = cv2.VideoCapture(selected_path)
        else:
            st.warning("⚠️ No videos found in video/raw/")

    if cap and cap.isOpened():
        st.info("Processing video file...")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.success("Video finished.")
                break
            
            # Use the original full-render function for files.
            pred_frame, render_frame = utils.predict_and_render(model, model_file, frame)
            
            video_frame_placeholder.image(pred_frame, caption="Input Video", channels="BGR")
            render_frame_placeholder.image(render_frame, caption="3D Pose", channels="BGR")
            
            time.sleep(0.01)
        cap.release()

elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload a stereo image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pred_frame, render_frame = utils.predict_and_render(model, model_file, frame_bgr)
        video_frame_placeholder.image(pred_frame, caption="Input Image", channels="BGR")
        render_frame_placeholder.image(render_frame, caption="3D Pose", channels="BGR")