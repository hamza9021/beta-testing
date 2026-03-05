# ─────────────────────────────────────────────
#  app.py  —  Flask Web Server (client-side camera edition)
# ─────────────────────────────────────────────
#
#  Architecture:
#    Browser grabs webcam via getUserMedia()
#    → POSTs JPEG frames to POST /detect
#    → Server runs YOLO, returns annotated JPEG
#    → Browser draws result on a <canvas>
#
#  Run locally:
#    python app.py
#  Then open:
#    http://127.0.0.1:5000
# ─────────────────────────────────────────────

from flask import Flask, Response, render_template, jsonify, request
from detector_web import WebDetector

app      = Flask(__name__)
detector = WebDetector()


# ── Detection endpoint ────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """
    Accepts a raw JPEG in the request body.
    Returns the annotated JPEG as image/jpeg.
    Detection stats are packed into response headers for the JS to read.
    """
    jpeg_bytes = request.data
    if not jpeg_bytes:
        return "No image data", 400

    out_jpeg, stats = detector.process_frame(jpeg_bytes)
    if not out_jpeg:
        return "Failed to process frame", 500

    resp = Response(out_jpeg, mimetype="image/jpeg")
    # Embed lightweight stats in headers so the JS can read without a
    # separate /api/stats round-trip
    resp.headers["X-FPS"]        = str(stats["fps"])
    resp.headers["X-Total"]      = str(stats["total"])
    resp.headers["X-Detections"] = str(stats["detections"])  # repr string; JS parses it
    resp.headers["Access-Control-Expose-Headers"] = "X-FPS, X-Total, X-Detections"
    return resp


# ── Config endpoint ───────────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "confidence" in data:
            detector.set_confidence(float(data["confidence"]))
        return jsonify({"status": "ok", "confidence": detector.conf})
    return jsonify({"confidence": detector.conf})


# ── Main page ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  YOLOv8 Device Detector — Web Edition (client-cam)")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
