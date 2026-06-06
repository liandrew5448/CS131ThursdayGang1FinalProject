import base64
import io
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__)

DATABASE_PATH = os.environ.get("POTHOLE_DATABASE", "cloud_potholes.db")
IMAGE_DIR = os.environ.get("POTHOLE_IMAGE_DIR", "cloud_pothole_images")
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("POTHOLE_API_KEY")
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_BUCKET_NAME = os.environ.get("GCP_BUCKET")
USE_GOOGLE_CLOUD = bool(GCP_BUCKET_NAME)

if USE_GOOGLE_CLOUD:
    from google.cloud import firestore
    from google.cloud import storage

    firestore_client = firestore.Client(project=GCP_PROJECT_ID)
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    storage_bucket = storage_client.bucket(GCP_BUCKET_NAME)

MAP_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pothole Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; color: #172126; font-family: Arial, sans-serif; }
        header {
            display: flex; align-items: center; justify-content: space-between;
            min-height: 58px; padding: 12px 18px; background: #f8faf9;
            border-bottom: 1px solid #d8dfdc;
        }
        h1 { margin: 0; font-size: 20px; }
        #summary { color: #52615d; font-size: 14px; }
        #map { height: calc(100vh - 58px); width: 100%; }
        .popup-image { display: block; width: 240px; max-width: 100%; margin-top: 8px; }
        .popup-title { font-weight: bold; }
        .popup-meta { margin-top: 4px; color: #52615d; }
    </style>
</head>
<body>
    <header>
        <h1>Pothole Map</h1>
        <div id="summary">Loading detections...</div>
    </header>
    <main id="map"></main>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const map = L.map("map").setView([33.9755, -117.3261], 16);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 20,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        }).addTo(map);

        function escapeHtml(value) {
            return String(value ?? "").replace(/[&<>"']/g, character => ({
                "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
            })[character]);
        }

        fetch("/api/potholes")
            .then(response => response.json())
            .then(data => {
                const potholes = data.potholes;
                document.getElementById("summary").textContent =
                    potholes.length + (potholes.length === 1 ? " pothole" : " potholes");

                const markers = potholes.map(pothole => {
                    const popup = `
                        <div class="popup-title">Pothole detection</div>
                        <div class="popup-meta">${escapeHtml(pothole.timestamp)}</div>
                        <div class="popup-meta">Confidence: ${Number(pothole.max_confidence).toFixed(2)}</div>
                        ${pothole.image_url ? `<img class="popup-image" src="${escapeHtml(pothole.image_url)}" alt="Detected pothole">` : ""}
                    `;
                    return L.marker([pothole.lat, pothole.lng]).addTo(map).bindPopup(popup);
                });

                if (markers.length) {
                    map.fitBounds(L.featureGroup(markers).getBounds().pad(0.2), { maxZoom: 18 });
                }
            })
            .catch(error => {
                document.getElementById("summary").textContent = "Could not load potholes";
                console.error(error);
            });
    </script>
</body>
</html>
"""


def get_database():
    database = sqlite3.connect(DATABASE_PATH)
    database.row_factory = sqlite3.Row
    return database


def initialize_database():
    with get_database() as database:
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS potholes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                accuracy REAL,
                count INTEGER NOT NULL,
                max_confidence REAL NOT NULL,
                source TEXT NOT NULL,
                image_filename TEXT
            )
            """
        )


def serialize_pothole(pothole):
    pothole = dict(pothole)
    pothole["image_url"] = (
        f"/images/{pothole.pop('image_filename')}"
        if pothole["image_filename"]
        else None
    )
    return pothole


@app.route("/")
def map_page():
    return MAP_PAGE


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/images/<path:filename>")
def images(filename):
    if USE_GOOGLE_CLOUD:
        blob = storage_bucket.blob(filename)
        if not blob.exists():
            return jsonify({"status": "error", "message": "image not found"}), 404
        return send_file(io.BytesIO(blob.download_as_bytes()), mimetype="image/jpeg")
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/api/potholes", methods=["GET"])
def list_potholes():
    if USE_GOOGLE_CLOUD:
        documents = (
            firestore_client.collection("potholes")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )
        potholes = []
        for document in documents:
            pothole = document.to_dict()
            pothole["id"] = document.id
            potholes.append(serialize_pothole(pothole))
        return jsonify({"potholes": potholes})

    with get_database() as database:
        rows = database.execute(
            "SELECT * FROM potholes ORDER BY timestamp DESC"
        ).fetchall()
    return jsonify({"potholes": [serialize_pothole(row) for row in rows]})


@app.route("/api/potholes", methods=["POST"])
def create_pothole():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"status": "error", "message": "unauthorized"}), 401

    event = request.get_json() or {}
    required_fields = ("lat", "lng", "max_confidence")
    missing_fields = [field for field in required_fields if event.get(field) is None]
    if missing_fields:
        return jsonify({"status": "error", "missing_fields": missing_fields}), 400

    image_filename = None
    image_base64 = event.get("image")
    if image_base64:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except ValueError:
            return jsonify({"status": "error", "message": "image is not valid base64"}), 400

        image_filename = f"pothole_{uuid.uuid4().hex}.jpg"
        if USE_GOOGLE_CLOUD:
            storage_bucket.blob(image_filename).upload_from_string(
                image_bytes,
                content_type="image/jpeg",
            )
        else:
            os.makedirs(IMAGE_DIR, exist_ok=True)
            with open(os.path.join(IMAGE_DIR, image_filename), "wb") as image_file:
                image_file.write(image_bytes)

    timestamp = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
    pothole = {
        "timestamp": timestamp,
        "lat": event["lat"],
        "lng": event["lng"],
        "accuracy": event.get("accuracy"),
        "count": event.get("count", 1),
        "max_confidence": event["max_confidence"],
        "source": event.get("source", "camera"),
        "image_filename": image_filename,
    }

    if USE_GOOGLE_CLOUD:
        _, document = firestore_client.collection("potholes").add(pothole)
        return jsonify({"status": "stored", "id": document.id}), 201

    with get_database() as database:
        cursor = database.execute(
            """
            INSERT INTO potholes (
                timestamp, lat, lng, accuracy, count, max_confidence, source, image_filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pothole["timestamp"],
                pothole["lat"],
                pothole["lng"],
                pothole["accuracy"],
                pothole["count"],
                pothole["max_confidence"],
                pothole["source"],
                pothole["image_filename"],
            ),
        )
        pothole_id = cursor.lastrowid

    return jsonify({"status": "stored", "id": pothole_id}), 201


if not USE_GOOGLE_CLOUD:
    initialize_database()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
