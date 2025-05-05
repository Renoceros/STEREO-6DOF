import cv2
import os
import numpy as np

# Define the face names
face_names = ["front", "back", "top", "bottom", "left", "right"]

# Create the ArUco directory if it doesn't exist
output_dir = "ArUco"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set ArUco parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size = 875  # Size of the markers in pixels
border_bits = 1    # Border around the marker

# Generate and save markers
for i, name in enumerate(face_names):
    # Create marker image (using the index as the marker ID)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size, borderBits=border_bits)
    
    # Convert to color (BGR) by stacking the single channel
    marker_img_color = cv2.merge([marker_img]*3)
    
    # Create a white border around the marker
    bordered_img = np.ones((marker_size + 70, marker_size + 70, 3), dtype=np.uint8) * 255
    bordered_img[35:-35, 35:-35] = marker_img_color
    
    # Save the marker
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(output_path, bordered_img)
    print(f"Generated and saved: {output_path}")

print("All markers generated successfully!")