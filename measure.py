#!/usr/bin/env python3
import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils.perspective import order_points   # ← add this import
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
# python3 matecvpi.py

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# === USER PARAMETERS ===
KNOWN_WIDTH = 8.5  # cm, width of your reference object
pixel_per_metric = None
cam_idx = 0
cap = cv2.VideoCapture(cam_idx)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue

        # compute the rotated bounding box
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = order_points(box)            # ← use the explicitly imported function

        cv2.drawContours(frame, [box.astype("int")], -1,
                         (0, 255, 0), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        for (x, y) in [(tltrX, tltrY), (blbrX, blbrY),
                       (tlblX, tlblY), (trbrX, trbrY)]:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixel_per_metric is None:
            pixel_per_metric = dB / KNOWN_WIDTH

        dimA = dA / pixel_per_metric
        dimB = dB / pixel_per_metric

        cv2.putText(frame, "{:.1f}cm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2)
        cv2.putText(frame, "{:.1f}cm".format(dimB),
                    (int(trbrX + 10), int(trbrY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255), 2)

    cv2.imshow("Real-time Measurement", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
