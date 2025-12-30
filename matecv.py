import cv2
import os
import time
import sys
import select
from ultralytics import YOLO
# how to run
# sudo apt update
# sudo apt install -y python3-pip python3-opencv
# pip3 install ultralytics
# scp app.py yolov8n.pt pi@10.42.0.2:/home/pi/
# sudo raspi-config
# → Interfaces → Camera → Enable
# reboot
# ssh pi@10.42.0.2
# enter password
# cd ~/        location of matecvpi.py
# python3 matecvpi.pyprint("Commands:")
print("    r : record video")
print("    q : quit")

# Load YOLOv8 model (change to your own model if needed)
model = YOLO("yolov8n.pt")  # or a custom .pt file

# Camera setup
camera_index = 0
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    print("Error - could not open video device.")
    sys.exit(1)

# Frame size
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
MJPG = cv2.VideoWriter_fourcc(*'MJPG')

def non_blocking_input(timeout=0.0):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline().strip().lower()
    return None

def record_video():
    try:
        duration = float(input("Enter recording duration in seconds: "))
    except ValueError:
        print("Invalid duration.")
        return

    output_dir = os.path.expanduser("~/mbhagatw/matecv/assets/cameraoutputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"recorded_{timestamp}.avi")

    out = cv2.VideoWriter(filename, MJPG, 20, (width, height))
    start_time = time.time()

    print(f"Recording started. Saving to {filename}")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Camera read error.")
            break

        # Inference and annotation
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        cv2.imshow("Recording", annotated_frame)

        if time.time() - start_time >= duration:
            break

        cmd = non_blocking_input(0.0)
        if cmd == 's':
            print("Stopped recording early.")
            break

        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cv2.destroyWindow("Recording")
    print("Recording saved.")

# Main loop
cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Live Detection", annotated_frame)

    cmd = non_blocking_input(0.1)
    if cmd == 'q':
        print("Exiting.")
        break
    elif cmd == 'r':
        record_video()

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
