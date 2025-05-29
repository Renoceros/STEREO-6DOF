# %%
# Updated: aruco_dataset_builder_splitter.py
import os
import cv2
import csv
import json
import numpy as np
import shutil
import random
from tqdm import tqdm
from datetime import datetime
from aruco_box_pose_estimation import CubePoseEstimator
import time
# %% Configuration
BATCH_ID = 3
FRAME_STEP = 5  # Skip every N frames to reduce redundancy
VAL_RATIO = 0.1
TEST_RATIO = 0.2
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO

VIDEO_PATH = "~/SKRIPSI/SCRIPTS/video/raw/ArUcoCom.avi"
BATCH_FOLDER = f"~/SKRIPSI/SCRIPTS/dataset/batch{BATCH_ID}"
UNSPLIT_FOLDER = os.path.join(BATCH_FOLDER, "unsplit")
IMAGES_FOLDER = os.path.join(UNSPLIT_FOLDER, "images")
LABELS_PATH = os.path.join(UNSPLIT_FOLDER, "labels.csv")

os.makedirs(IMAGES_FOLDER, exist_ok=True)

# %%
# Initialize
cap = cv2.VideoCapture(VIDEO_PATH)
estimator = CubePoseEstimator()
frame_idx = 0
write_idx = 0
now = []
now.append(time.time())
with open(LABELS_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'x', 'y', 'z', 'pitch', 'roll', 'yaw'])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP != 0:
                frame_idx += 1
                continue

            cube_rvec, cube_tvec = estimator.process_frame(frame)
            if cube_rvec is not None:
                rmat, _ = cv2.Rodrigues(cube_rvec)
                pitch = np.degrees(np.arcsin(-rmat[2, 0])) / 360.0
                roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])) / 360.0
                yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])) / 360.0

                fname = f"frame_{write_idx:04d}.png"
                image_path = os.path.join(IMAGES_FOLDER, fname)
                cv2.imwrite(image_path, frame)

                writer.writerow([
                    fname,
                    float(cube_tvec[0][0]),
                    float(cube_tvec[1][0]),
                    float(cube_tvec[2][0]),
                    pitch,
                    roll,
                    yaw
                ])

                write_idx += 1

            frame_idx += 1
    finally:
        cap.release()
        print("Finished reading video and writing labels.")
        print(f"Time took : {int(now[1]-now[0])}s")
# %%
# Load labels for split
with open(LABELS_PATH, 'r') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    rows = reader[1:]

random.seed(42)
random.shuffle(rows)

total = len(rows)
train_end = int(TRAIN_RATIO * total)
val_end = train_end + int(VAL_RATIO * total)

# First split by chunks, then shuffle each set to avoid overlap
train = rows[:train_end]
val = rows[train_end:val_end]
test = rows[val_end:]

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

# %%
# Create folders
for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(BATCH_FOLDER, subset, 'images'), exist_ok=True)
now = []
now.append(time.time())
# Copy images + write labels
for subset_name, data in zip(['train', 'val', 'test'], [train, val, test]):
    subset_folder = os.path.join(BATCH_FOLDER, subset_name)
    csv_path = os.path.join(subset_folder, 'labels.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)
            src_img = os.path.join(IMAGES_FOLDER, row[0])
            dst_img = os.path.join(subset_folder, 'images', row[0])
            shutil.copy(src_img, dst_img)

print("✅ Dataset split complete.")
print(f"Time took : {int(now[1]-now[0])}s")
# %%
# Write metadata
sample_frame = cv2.imread(os.path.join(IMAGES_FOLDER, "frame_0000.png"))
metadata_path = os.path.join(BATCH_FOLDER, 'metadata.md')
with open(metadata_path, 'w') as f:
    f.write(f"# Batch {BATCH_ID} Metadata\n")
    f.write(f"- Date: {datetime.now().isoformat()}\n")
    f.write(f"- Total Frames: {len(rows)}\n")
    f.write(f"- Frame Step: {FRAME_STEP}\n")
    f.write(f"- Train: {len(train)} | Val: {len(val)} | Test: {len(test)}\n")
    f.write(f"- Image Resolution: {sample_frame.shape[1]}x{sample_frame.shape[0]}\n")
    f.write(f"- ArUco-based 6DoF Pose Estimation\n")
    f.write(f"- Pose Normalization: Euler angles divided by 360\n")
    f.write(f"- Cube Size: {estimator.CUBE_SIZE if hasattr(estimator, 'CUBE_SIZE') else 0.08}m\n")

print(f"📝 Metadata saved to {metadata_path}")
