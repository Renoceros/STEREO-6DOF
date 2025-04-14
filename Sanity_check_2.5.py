# Sanity_check_2.5.py
import cv2
import os
import config as c
import utils.stereo_utils as su

def main():
    batch_num = 1  # Modify this as needed to select the batch
    left_video_path = os.path.join(c.vid_preprocessed, f'BATCH_{batch_num}', 'Left.avi')
    right_video_path = os.path.join(c.vid_preprocessed, f'BATCH_{batch_num}', 'Right.avi')

    # Open Left and Right videos
    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or both video files.")
        return

    # Get video properties (assuming both videos have the same resolution)
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {frame_width}x{frame_height}, FPS: {fps}")

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: Could not read frame from one of the videos.")
            break

        # Display the frames
        combined_frame = cv2.hconcat([frame_left, frame_right])  # Combine the two frames horizontally
        cv2.imshow("Left and Right Videos", combined_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start, start_str = su.Current()
    print("Start Time : "+start_str)
    main()
    end, end_str = su.Current()
    print("End Time : "+end_str)
    print("Duration : "+str(end-start))