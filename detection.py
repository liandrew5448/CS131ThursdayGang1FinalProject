import base64
import os
import platform
import json
import ssl
import time
import urllib.error
import urllib.request

import cv2
import torch
from ultralytics import YOLO


CAMERA_INDEXES = (0, 1, 2, 3)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMG_SIZE = 320
CONFIDENCE = 0.7

# Increase this if the preview is still too choppy. Lower it for quicker box updates.
DETECT_EVERY = 2
POTHOLE_SERVER_URL = os.environ.get(
    "POTHOLE_SERVER_URL",
    "https://127.0.0.1:5002/pothole",
)
POTHOLE_LOG_COOLDOWN_SECONDS = 5


def pick_model_path():
    if os.path.exists("best.engine"):
        return "best.engine"
    return "best.pt"


def pick_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_info(device):
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("MPS available:", torch.backends.mps.is_available())

    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    elif platform.system() == "Linux":
        print(
            "Warning: running on CPU. On Jetson, this usually means the installed "
            "PyTorch wheel does not match JetPack/CUDA, or CUDA is not configured."
        )

    print("Using device:", device)


def configure_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera():
    is_mac = platform.system() == "Darwin"

    for camera_index in CAMERA_INDEXES:
        if is_mac:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(camera_index)

        configure_camera(cap)

        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Using camera index:", camera_index)
                return cap
            time.sleep(0.1)

        cap.release()

    return None


def send_pothole_event(count, max_confidence, image):
    image_base64 = None
    success, encoded_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if success:
        image_base64 = base64.b64encode(encoded_image).decode("utf-8")

    payload = {
        "source": "camera",
        "count": count,
        "max_confidence": max_confidence,
        "image": image_base64,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        POTHOLE_SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(request, timeout=1, context=context) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        print("Could not log pothole:", error)


def get_detection_summary(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, None

    confidences = boxes.conf
    max_confidence = float(confidences.max().item()) if confidences is not None else None
    return len(boxes), max_confidence


def main():
    model_path = pick_model_path()
    device = pick_device()
    print_device_info(device)
    print("Using model:", model_path)
    print("Pothole server:", POTHOLE_SERVER_URL)

    model = YOLO(model_path)

    cap = open_camera()
    if cap is None:
        print("Camera did not open or did not return frames")
        return

    frame_count = 0
    last_annotated_frame = None
    last_fps_time = time.time()
    last_pothole_log_time = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read camera frame")
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_count % DETECT_EVERY == 0:
            if model_path.endswith(".engine"):
                results = model(frame, imgsz=IMG_SIZE, conf=CONFIDENCE, verbose=False)
            else:
                results = model(
                    frame,
                    imgsz=IMG_SIZE,
                    conf=CONFIDENCE,
                    device=device,
                    half=torch.cuda.is_available(),
                    verbose=False,
                )
            last_annotated_frame = results[0].plot()

            detection_count, max_confidence = get_detection_summary(results[0])
            now = time.time()
            if (
                detection_count > 0
                and now - last_pothole_log_time >= POTHOLE_LOG_COOLDOWN_SECONDS
            ):
                send_pothole_event(detection_count, max_confidence, last_annotated_frame)
                last_pothole_log_time = now

        frame_count += 1

        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            last_fps_time = now

        output_frame = last_annotated_frame if last_annotated_frame is not None else frame
        cv2.putText(
            output_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
