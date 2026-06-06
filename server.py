import base64
import csv
import json
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone

from flask import Flask, request, jsonify

app = Flask(__name__)

latest_gps = None
pothole_log = []
last_pothole_time = None
POTHOLE_LOG_FILE = "potholes.csv"
POTHOLE_IMAGE_DIR = "pothole_images"
DUPLICATE_TIME_SECONDS = 5
CLOUD_SERVER_URL = os.environ.get("CLOUD_SERVER_URL")
CLOUD_API_KEY = os.environ.get("CLOUD_API_KEY")
CLOUD_MIN_CONFIDENCE = float(os.environ.get("CLOUD_MIN_CONFIDENCE", "0.70"))

HTML_PAGE = """
<!DOCTYPE html>
<html>
<body>

<h2>Phone GPS Sender</h2>
<p id="status">Waiting for GPS...</p>

<script>

function sendGPS(position) {
    const gps = {
        lat: position.coords.latitude,
        lng: position.coords.longitude,
        accuracy: position.coords.accuracy
    };

    fetch("/gps", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(gps)
    }).then(function(response) {
        if (!response.ok) {
            throw new Error("Server returned " + response.status);
        }

        document.getElementById("status").innerText =
            "GPS Sent: " + gps.lat + ", " + gps.lng +
            " (accuracy " + Math.round(gps.accuracy) + "m)";
    }).catch(function(error) {
        document.getElementById("status").innerText =
            "GPS found, but failed to send to server: " + error.message;
    });
}

function gpsError(error) {
    let message = error.message;

    if (!window.isSecureContext) {
        message =
            "Browser blocked GPS because this page is not HTTPS. " +
            "Use localhost, HTTPS, or allow insecure origins for testing.";
    }

    document.getElementById("status").innerText = "GPS error: " + message;
    console.error(error);
}

navigator.geolocation.watchPosition(
    sendGPS,
    gpsError,
    {
        enableHighAccuracy: true
    }
);

</script>

</body>
</html>
"""

@app.route("/")
def home():
    return HTML_PAGE

@app.route("/gps", methods=["POST"])
def gps():

    global latest_gps

    latest_gps = request.get_json()

    print("GPS UPDATE:", latest_gps)

    return jsonify({"status": "received"})

@app.route("/latest")
def latest():
    return jsonify({"gps": latest_gps})

def get_cloud_skip_reason(log_entry):
    if not CLOUD_SERVER_URL:
        return "cloud_not_configured"
    if log_entry["lat"] is None or log_entry["lng"] is None:
        return "missing_gps"
    if not log_entry["image_path"]:
        return "missing_image"
    if (log_entry["max_confidence"] or 0) < CLOUD_MIN_CONFIDENCE:
        return "low_confidence"
    return None

def send_to_cloud(log_entry, image_base64):
    cloud_event = dict(log_entry)
    cloud_event["image"] = image_base64
    cloud_event.pop("image_path", None)
    data = json.dumps(cloud_event).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if CLOUD_API_KEY:
        headers["X-API-Key"] = CLOUD_API_KEY

    cloud_request = urllib.request.Request(
        CLOUD_SERVER_URL,
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(cloud_request, timeout=5, context=context) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        print("CLOUD UPLOAD:", response_data)
        return {"status": "uploaded", "response": response_data}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        print("CLOUD UPLOAD FAILED:", error)
        return {"status": "failed", "error": str(error)}

@app.route("/pothole", methods=["POST"])
def pothole():
    global last_pothole_time

    event = request.get_json() or {}
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    safe_timestamp = timestamp.replace(":", "-").replace("+", "_")
    image_path = None

    if last_pothole_time is not None:
        seconds_since_last = (now - last_pothole_time).total_seconds()
        if seconds_since_last < DUPLICATE_TIME_SECONDS:
            print(
                "POTHOLE SKIPPED:",
                f"last log was {seconds_since_last:.1f}s ago",
            )
            return jsonify({
                "status": "skipped_duplicate_time",
                "seconds_since_last": seconds_since_last,
                "duplicate_time_seconds": DUPLICATE_TIME_SECONDS,
            })

    image_base64 = event.get("image")
    if image_base64:
        os.makedirs(POTHOLE_IMAGE_DIR, exist_ok=True)
        image_path = os.path.join(POTHOLE_IMAGE_DIR, f"pothole_{safe_timestamp}.jpg")

        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_base64))

    log_entry = {
        "timestamp": timestamp,
        "lat": latest_gps.get("lat") if latest_gps else None,
        "lng": latest_gps.get("lng") if latest_gps else None,
        "accuracy": latest_gps.get("accuracy") if latest_gps else None,
        "count": event.get("count", 0),
        "max_confidence": event.get("max_confidence"),
        "source": event.get("source", "camera"),
        "image_path": image_path,
    }

    pothole_log.append(log_entry)
    last_pothole_time = now
    write_header = not os.path.exists(POTHOLE_LOG_FILE)

    with open(POTHOLE_LOG_FILE, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(log_entry)

    print("POTHOLE DETECTED:", log_entry)

    cloud_skip_reason = get_cloud_skip_reason(log_entry)
    if cloud_skip_reason:
        cloud_result = {"status": "skipped", "reason": cloud_skip_reason}
        print("CLOUD UPLOAD SKIPPED:", cloud_skip_reason)
    else:
        cloud_result = send_to_cloud(log_entry, image_base64)

    return jsonify({"status": "logged", "event": log_entry, "cloud": cloud_result})

@app.route("/potholes")
def potholes():
    return jsonify({"potholes": pothole_log})


if __name__ == "__main__":
    print("Cloud server:", CLOUD_SERVER_URL or "not configured")
    print("Cloud minimum confidence:", CLOUD_MIN_CONFIDENCE)
    app.run(host="0.0.0.0", port=5002, ssl_context="adhoc")
