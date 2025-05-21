import cv2
import os
import shutil
import subprocess
import sys
from shutil import which

# Directories
TEMP_DIR = os.path.abspath("video/temp")
OUTPUT_DIR = os.path.abspath("video/raw")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "ArUco2.mp4")

# Ensure output directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera 0")

frame_count = 0
recording = False
print("Press [SPACE] to toggle recording, [q] to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_resized = cv2.resize(frame, (1280, 480))
    status_text = "Recording..." if recording else "Paused"

    # Display status on the frame
    cv2.putText(frame_resized, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255) if recording else (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Capture - SPACE to Record, Q to Quit", frame_resized)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE toggles recording
        recording = not recording
        print("Recording started." if recording else "Recording stopped.")

    elif key == ord('q'):  # Quit
        print("Exiting...")
        break

    if recording:
        filename = os.path.join(TEMP_DIR, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(filename, frame_resized)
        frame_count += 1

# Cleanup camera
cap.release()
cv2.destroyAllWindows()
# frame_count = 10586
# If frames were captured, encode them
if frame_count > 0:
    print(f"Recorded {frame_count} frames...")
    
    if which("ffmpeg") is None:
        print("⚠️  FFmpeg not found. Please install it with: sudo apt install ffmpeg")
        print(f"Your frames are saved in: {TEMP_DIR}")
        sys.exit(1)

    print("Encoding video with FFmpeg...")
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", os.path.join(TEMP_DIR, "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        OUTPUT_VIDEO
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"✅ Video saved to: {OUTPUT_VIDEO}")
        shutil.rmtree(TEMP_DIR)
        print("🗑️ Temporary frames deleted.")
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg encoding failed:", e)
        print(f"Frames preserved at: {TEMP_DIR}")
else:
    print("No frames recorded. Skipping video encoding.")
