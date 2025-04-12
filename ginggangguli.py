import numpy as np

def compute_focal_length(fov_deg, width_px):
    fov_rad = np.deg2rad(fov_deg)
    f = width_px / (2 * np.tan(fov_rad / 2))
    return f

# Assuming final rectified width is 1920px
focal_length_px = compute_focal_length(130, 1100)
print(f"Focal Length (px): {focal_length_px:.2f}")
