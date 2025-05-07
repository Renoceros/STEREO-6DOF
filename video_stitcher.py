import cv2

VID_1 = "video/raw/ArUco.avi"
VID_2 = "video/raw/ArUco2.avi"
VID_RESULT = "video/raw/ArUcoCom.avi"

# Open the two input videos
cap1 = cv2.VideoCapture(VID_1)
cap2 = cv2.VideoCapture(VID_2)

# Check both videos opened successfully
if not cap1.isOpened():
    raise IOError(f"Failed to open {VID_1}")
if not cap2.isOpened():
    raise IOError(f"Failed to open {VID_2}")

# Get video properties from the first video
fps = cap1.get(cv2.CAP_PROP_FPS)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup the output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'MJPG', 'mp4v', etc.
out = cv2.VideoWriter(VID_RESULT, fourcc, fps, (width, height))

def write_video_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

# Write frames from both videos sequentially
write_video_frames(cap1)
write_video_frames(cap2)

# Release everything
cap1.release()
cap2.release()
out.release()

print(f"Stitched video saved as: {VID_RESULT}")
