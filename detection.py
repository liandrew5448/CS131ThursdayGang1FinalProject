import base64
import os
import platform
import json
import ssl
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import torch
from ultralytics import YOLO

CAMERA_INDEXES = (0, 1)
PREFERRED_CAMERA_INDEX = os.environ.get("CAMERA_INDEX")
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
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
PREVIEW_HOST = os.environ.get("PREVIEW_HOST", "0.0.0.0")
PREVIEW_PORT = int(os.environ.get("PREVIEW_PORT", "8080"))
PREVIEW_JPEG_QUALITY = 70

latest_preview_frame = None
preview_frame_condition = threading.Condition()


def pick_model_path():
    if os.path.exists("best.engine"):
        return "best.engine"
    return "best1500.pt"


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
    if platform.system() == "Darwin":
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera():
    is_mac = platform.system() == "Darwin"
    camera_indexes = CAMERA_INDEXES
    if PREFERRED_CAMERA_INDEX is not None:
        camera_indexes = (int(PREFERRED_CAMERA_INDEX),)

    for camera_index in camera_indexes:
        backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY) if is_mac else (cv2.CAP_ANY,)
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            configure_camera(cap)

            if not cap.isOpened():
                cap.release()
                continue

            backend_name = "AVFoundation" if backend == cv2.CAP_AVFOUNDATION else "default"
            print(
                "Waiting for camera index",
                camera_index,
                f"({backend_name}) to return frames...",
            )
            max_attempts = 30 if is_mac else 5
            for _ in range(max_attempts):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print("Using camera index:", camera_index, "backend:", backend_name)
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


def update_preview_frame(frame):
    global latest_preview_frame

    success, encoded_frame = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY],
    )
    if not success:
        return

    with preview_frame_condition:
        latest_preview_frame = encoded_frame.tobytes()
        preview_frame_condition.notify_all()


class PreviewRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""<!DOCTYPE html>
<html>
<head><title>Jetson Pothole Detection Preview</title></head>
<body style="margin:0;background:#111;color:#fff;font-family:Arial,sans-serif">
<h2 style="padding:12px 16px;margin:0">Jetson Pothole Detection Preview</h2>
<img src="/video_feed" alt="Live pothole detection preview"
     style="display:block;max-width:100%;height:auto">
</body>
</html>"""
            )
            return

        if self.path != "/video_feed":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with preview_frame_condition:
                    preview_frame_condition.wait(timeout=1)
                    frame = latest_preview_frame

                if frame is None:
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        return


def start_preview_server():
    server = ThreadingHTTPServer((PREVIEW_HOST, PREVIEW_PORT), PreviewRequestHandler)
    preview_thread = threading.Thread(target=server.serve_forever, daemon=True)
    preview_thread.start()
    print(f"Browser preview: http://JETSON_IP:{PREVIEW_PORT}/")


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
    print("Headless mode:", HEADLESS)
    start_preview_server()

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
        update_preview_frame(output_frame)

        if not HEADLESS:
            cv2.imshow("YOLO Detection", output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
