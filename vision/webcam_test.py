"""Step 2: verify the USB webcam works with OpenCV.

Setup (PowerShell):
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install opencv-python numpy

Run:
    python webcam_test.py                              # camera index 0
    python webcam_test.py 1                            # try index 1 if 0 is your laptop camera
    python webcam_test.py http://PHONE_IP:8080/video   # Android "IP Webcam" phone stream

Keys: q = quit, s = save snapshot (snapshots/empty_scene_N.jpg)
Use saved snapshots later as the empty-scene reference for calibration.
"""
import os
import sys
import time

import cv2

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"
CAM_INDEX = int(SOURCE) if SOURCE.isdigit() else SOURCE
WIDTH, HEIGHT = 1280, 720   # 720p is plenty for a 40x40 cm model and much faster than 1080p

# CAP_DSHOW opens faster and is more reliable on Windows
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) if isinstance(CAM_INDEX, int) and os.name == "nt" else cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    sys.exit(f"Could not open {CAM_INDEX}. Try another index (0, 1, 2) or check the phone URL and Wi-Fi.")

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera {CAM_INDEX} opened at {actual_w}x{actual_h}")

os.makedirs("snapshots", exist_ok=True)
snap_count = 0
frames, t0, fps = 0, time.time(), 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame read failed")
        break

    frames += 1
    if time.time() - t0 >= 1.0:
        fps = frames / (time.time() - t0)
        frames, t0 = 0, time.time()

    # Mean brightness helps you spot lighting changes, which affect image differencing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    cv2.putText(frame, f"{actual_w}x{actual_h}  {fps:.1f} fps  brightness {brightness:.0f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Webcam test (q quit, s snapshot)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        snap_count += 1
        path = f"snapshots/empty_scene_{snap_count}.jpg"
        cv2.imwrite(path, frame)
        print("Saved", path)

cap.release()
cv2.destroyAllWindows()
