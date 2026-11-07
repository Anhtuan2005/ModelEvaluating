
import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture("Intrusion.mp4")

assert cap.isOpened(), "Cannot open video"

times = []

for i in range(60):
    ok, frame = cap.read()
    if not ok:
        break

    start = time.perf_counter()
    model.predict(
        frame,
        classes=[0],
        conf=0.25,
        device="cpu",
        verbose=False
    )
    times.append(time.perf_counter() - start)

cap.release()

print(f"Frames tested: {len(times)}")
if times:
    print(f"Average inference FPS: {len(times)/sum(times):.2f}")
