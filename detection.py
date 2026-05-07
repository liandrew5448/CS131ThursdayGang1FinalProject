
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera did not open")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        continue

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()