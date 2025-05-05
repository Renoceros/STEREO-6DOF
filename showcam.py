import cv2
import utility.stereo_utils as su

# Open the camera
cap, w, h = su.OpenCam(0)
su.Ding()
if not cap or not cap.isOpened():
    print("Failed to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show the grayscale image
    cv2.imshow('Grayscale Camera Feed', gray)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
