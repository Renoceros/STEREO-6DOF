import cv2

input_path = "video/preprocessed/left.avi"
output_path = "augh.mp4" # Change to .mp4 if needed

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print(f"❌ Error: Couldn't open {input_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')#Use 'MP4V' for MP4

out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop at the first unreadable frame
    
    out.write(frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"✅ Converted {frame_count} frames...")

cap.release()
out.release()
print(f"🎉 Conversion complete! Saved as {output_path}")
print(f"{frame_width}")
