# capture_calibration_images.py
import cv2
import os
import numpy as np
import utility.stereo_utils as su  # Your usual utils

def draw_crosshairs(image, mode):
    h, w = image.shape[:2]
    overlay = image.copy()
    color = (0, 255, 0)
    thickness = 5

    if mode in ['LEFT', 'RIGHT']:
        # Center lines
        cv2.line(overlay, (w // 2, 0), (w // 2, h), color, thickness)
        cv2.line(overlay, (0, h // 2), (w, h // 2), color, thickness)
        # Extra horizontal guidelines
        cv2.line(overlay, (0, h // 8), (w, h // 8), color, thickness)
        cv2.line(overlay, (0, 7 * h // 8), (w, 7 * h // 8), color, thickness)

    elif mode == 'TOGETHER':
        # Center horizontal line
        cv2.line(overlay, (0, h // 2), (w, h // 2), color, thickness)
        # Top and bottom guidelines
        cv2.line(overlay, (0, h // 8), (w, h // 8), color, thickness)
        cv2.line(overlay, (0, 7 * h // 8), (w, 7 * h // 8), color, thickness)

    return overlay

def capture_images():
    Start,Startstr = su.Current()
    print("Start : "+Startstr)
    os.makedirs("capture/left", exist_ok=True)
    os.makedirs("capture/right", exist_ok=True)
    os.makedirs("capture/together", exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    su.Ding()
    image_count = 0
    mode = 'LEFT'  # Start with capturing LEFT images
    CamON,CamONstr = su.Current()
    print("Cam On : "+ CamONstr)
    print("Cam On time : "+str(CamON-Start))
    print("\nControls: [SPACE] Capture | [TAB] Switch Mode | [ESC] Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        height, width = frame.shape[:2]
        half_width = width // 2

        left = frame[:, :half_width]
        right = frame[:, half_width:]

        if mode == 'LEFT':
            display = draw_crosshairs(left.copy(), 'LEFT')
        elif mode == 'RIGHT':
            display = draw_crosshairs(right.copy(), 'RIGHT')
        else:  # mode == 'TOGETHER'
            right_flipped = cv2.flip(right, 1)  # Horizontal flip
            blend = cv2.addWeighted(left, 0.5, right_flipped, 0.5, 0)
            display = draw_crosshairs(blend, 'TOGETHER')

        # Show capture mode at top-left
        cv2.putText(display, f"MODE: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Calibration Images", display)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == 9:  # TAB to switch mode
            if mode == 'LEFT':
                mode = 'RIGHT'
            elif mode == 'RIGHT':
                mode = 'TOGETHER'
            else:
                mode = 'LEFT'
        elif key == 32:  # SPACE to capture
            su.Ding() #winsound make noise
            if mode == 'LEFT':
                save_path = f"development/left/left_{image_count:02d}.jpg"
                cv2.imwrite(save_path, left)
                print(f"Saved {save_path}")
            elif mode == 'RIGHT':
                save_path = f"development/right/right_{image_count:02d}.jpg"
                cv2.imwrite(save_path, right)
                print(f"Saved {save_path}")
            else:  # TOGETHER
                save_path = f"development/together/together_{image_count:02d}.jpg"
                combined = np.hstack((left, right))
                cv2.imwrite(save_path, combined)
                print(f"Saved {save_path}")

            image_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
