# New Implementation.py (Frontend)
import os
import cv2
import numpy as np
import streamlit as st
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from utility import slit_render_utils as utils

# ====== CONFIG ======
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "baked")

# ====== STREAMLIT UI ======
st.title("6D Pose Estimation - Implementation Demo")

mode = st.radio("Input Source", ["Stereo Webcam", "Upload Video", "Use Local Video", "Upload Image"])
model_file = st.selectbox("Select Model", utils.get_available_models(MODEL_DIR))
model = utils.load_model(MODEL_DIR, model_file)

video_frame = st.empty()
overlay_frame = st.empty()

class StereoVideoProcessor(utils.VideoProcessorBase):
    def __init__(self):
        self.result_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            pred_frame, render_frame = utils.predict_and_render(model, model_file, img)
            self.result_frame = render_frame
            return utils.to_av_frame(pred_frame)
        except Exception as e:
            print(f"[WebRTC Error] {e}")
            return frame

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
        st.warning("\u26a0\ufe0f No videos found in video/raw/")

elif mode == "Stereo Webcam":
    st.warning("\u26a0\ufe0f Using client webcam via WebRTC.")
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
        pred_frame, render_frame = utils.predict_and_render(model, model_file, frame)
        video_frame.image(pred_frame, caption="Input Frame", channels="RGB")
        overlay_frame.image(render_frame, caption="3D Pose", channels="RGB")

# ====== PROCESS LOOP ======
if cap is not None and cap.isOpened():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pred_frame, render_frame = utils.predict_and_render(model, model_file, frame)
        video_frame.image(pred_frame, channels="RGB")
        overlay_frame.image(render_frame, channels="RGB")
    cap.release()
else:
    if mode != "Upload Image":
        st.warning("\u23f3 Waiting for video input...")
