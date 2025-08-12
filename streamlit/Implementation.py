# Implementation.py
import os
import cv2
import numpy as np
import streamlit as st
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import time # Import the time module

# It's assumed that 'slit_render_utils' is in a directory named 'utility'
# and accessible from this script's location.
from utility import slit_render_utils as utils

# ====== CONFIGURATION ======
# Sets the directory to find the trained model files.
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

# Load the selected PyTorch model onto the appropriate device (GPU or CPU)
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


# ====== WEBRTC VIDEO PROCESSOR ======
class StereoVideoProcessor(utils.VideoProcessorBase):
    def __init__(self, model, model_file):
        self.model = model
        self.model_file = model_file

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            pred_frame, render_frame = utils.predict_and_render(self.model, self.model_file, img)
            h_pred, w_pred, _ = pred_frame.shape
            h_render, w_render, _ = render_frame.shape
            new_h_pred = int(h_pred * w_render / w_pred)
            resized_webcam = cv2.resize(pred_frame, (w_render, new_h_pred))
            combined_frame = np.vstack([resized_webcam, render_frame])
            return utils.to_av_frame(combined_frame)
        except Exception as e:
            print(f"[WebRTC Error] {e}")
            return utils.to_av_frame(img)


# ====== MAIN LOGIC FOR INPUT MODES ======
cap = None

if mode == "Stereo Webcam":
    webrtc_streamer(
        key="stereo-webrtc",
        video_processor_factory=lambda: StereoVideoProcessor(model=model, model_file=model_file),
        media_stream_constraints={"video": {"width": 1280, "height": 480}},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}),
    )

elif mode == "Upload Video":
    uploaded = st.file_uploader("Upload a stereo video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)

elif mode == "Use Local Video":
    raw_video_dir = os.path.join(os.path.dirname(__file__), "..", "video", "raw")
    available_videos = [f for f in os.listdir(raw_video_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    if available_videos:
        selected_video = st.selectbox("Select a video from raw/", available_videos)
        selected_path = os.path.join(raw_video_dir, selected_video)
        cap = cv2.VideoCapture(selected_path)
    else:
        st.warning("⚠️ No videos found in video/raw/")

elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload a stereo image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pred_frame, render_frame = utils.predict_and_render(model, model_file, frame_bgr)
        video_frame_placeholder.image(pred_frame, caption="Input Image", channels="BGR")
        render_frame_placeholder.image(render_frame, caption="3D Pose", channels="BGR")


# ====== PROCESSING LOOP FOR VIDEO FILES ======
if cap is not None and cap.isOpened():
    st.info("Processing video file...")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.success("Video finished.")
            break
        
        pred_frame, render_frame = utils.predict_and_render(model, model_file, frame)
        
        video_frame_placeholder.image(pred_frame, caption="Input Video", channels="BGR")
        render_frame_placeholder.image(render_frame, caption="3D Pose", channels="BGR")
        
        # FIX: Add a small delay to prevent overwhelming the network connection.
        # This effectively caps the framerate and allows the UI to keep up.
        time.sleep(0.01)

    cap.release()
else:
    if mode not in ["Stereo Webcam", "Upload Image"]:
        st.info("Waiting for video input...")

