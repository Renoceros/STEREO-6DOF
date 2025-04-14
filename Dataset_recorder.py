#Dataset_recorder.py
import cv2
import os
import numpy as np
import config as c
import utils.stereo_utils as su


def record_video(cap, output_dir, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(output_dir, fourcc, fps, frame_size, isColor=True)

    recording = False
    print("Press SPACE to start/stop recording. ESC to finish.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame from video source.")
            break

        cv2.imshow("Recording", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            recording = not recording
            print("Recording started!" if recording else "Recording stopped!")

        if recording:
            out_video.write(frame)

    out_video.release()


def split_video(input_video_path, output_left_path, output_right_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file {input_video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_left = cv2.VideoWriter(output_left_path, fourcc, fps, (frame_width // 2, frame_height))
    out_right = cv2.VideoWriter(output_right_path, fourcc, fps, (frame_width // 2, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_frame = frame[:, :frame_width // 2]
        right_frame = frame[:, frame_width // 2:]

        out_left.write(left_frame)
        out_right.write(right_frame)

    cap.release()
    out_left.release()
    out_right.release()


def main():
    existing_batches = [d for d in os.listdir(c.vid_preprocessed)
                        if os.path.isdir(os.path.join(c.vid_preprocessed, d)) and d.startswith('BATCH_')]
    batch_num = len(existing_batches)
    output_dir = os.path.join(c.vid_preprocessed, f'BATCH_{batch_num + 1}')
    os.makedirs(output_dir, exist_ok=True)

    temp_video_path = "video/temp/recorded_video.avi"
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open video capture.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback in case FPS is zero
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps}")

    record_video(cap, temp_video_path, fps, (frame_width, frame_height))

    cap.release()
    cv2.destroyAllWindows()

    print("Splitting video...")
    left_video_path = os.path.join(output_dir, 'Left.avi')
    right_video_path = os.path.join(output_dir, 'Right.avi')
    split_video(temp_video_path, left_video_path, right_video_path)

    os.remove(temp_video_path)
    print("Recording and splitting complete.")


if __name__ == "__main__":
    start, start_str = su.Current()
    print("Start Time : " + start_str)
    main()
    end, end_str = su.Current()
    print("End Time : " + end_str)
    print("Duration : " + str(end - start))
