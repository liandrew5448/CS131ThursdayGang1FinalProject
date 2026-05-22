import base64
import csv
import os
from datetime import datetime, timezone

from flask import Flask, request, jsonify

app = Flask(__name__)

latest_gps = None
pothole_log = []
last_pothole_time = None
POTHOLE_LOG_FILE = "potholes.csv"
POTHOLE_IMAGE_DIR = "pothole_images"
DUPLICATE_TIME_SECONDS = 10

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

    return jsonify({"status": "logged", "event": log_entry})

@app.route("/potholes")
def potholes():
    return jsonify({"potholes": pothole_log})


app.run(host="0.0.0.0", port=5002, ssl_context="adhoc")
