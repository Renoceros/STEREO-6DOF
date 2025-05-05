# generate_checkerboard.py
import cv2
import numpy as np
import os

# Settings (match calibration notebook)
SQUARE_SIZE_MM = 25  # 0.025m = 25mm
DOTS_PER_MM = 300 / 25.4  # 300ppi → mm
CHESSBOARD_SIZE = (9, 6)  # Internal corners
MARGIN_MM = 0  # White border

# Calculate dimensions
square_px = int(SQUARE_SIZE_MM * DOTS_PER_MM)
margin_px = int(MARGIN_MM * DOTS_PER_MM)
width = CHESSBOARD_SIZE[0] * square_px + 2 * margin_px
height = CHESSBOARD_SIZE[1] * square_px + 2 * margin_px

# Create white image
img = np.ones((height, width), dtype=np.uint8) * 255

# Draw checkerboard
for i in range(CHESSBOARD_SIZE[0] + 1):
    for j in range(CHESSBOARD_SIZE[1] + 1):
        if (i + j) % 2 == 0:
            x1 = margin_px + i * square_px
            y1 = margin_px + j * square_px
            img[y1:y1+square_px, x1:x1+square_px] = 0

# Save
os.makedirs("calibration_images", exist_ok=True)
output_path = "calibration_images/cb.png"
cv2.imwrite(output_path, img)

print(f"Checkerboard saved to {output_path}")
print(f"Physical size: {(width/DOTS_PER_MM)/10:.1f}cm x {(height/DOTS_PER_MM)/10:.1f}cm")
print(f"Recommended print size: {width/300:.2f}in x {height/300:.2f}in at 300ppi")