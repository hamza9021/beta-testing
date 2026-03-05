# ─────────────────────────────────────────────
#  app.py  —  Flask Web Server
# ─────────────────────────────────────────────
#
#  Run:
#    python app.py
#  Then open:
#    http://127.0.0.1:5000
# ─────────────────────────────────────────────

from flask import Flask, Response, render_template, jsonify, request
from detector_web import WebDetector
import time

app = Flask(__name__)

# Global detector instance (started lazily on first /video_feed request)
detector = WebDetector()
detector.start()   # start immediately so the stream is ready


# ── MJPEG frame generator ────────────────────────────────────────────────────
def _generate_frames():
    """Yields continuous multipart JPEG frames for the browser."""
    while True:
        frame = detector.get_frame()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        else:
            # Not ready yet — send a tiny sleep so we don't spin-burn CPU
            time.sleep(0.05)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(detector.get_stats())


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "confidence" in data:
            detector.set_confidence(float(data["confidence"]))
        return jsonify({"status": "ok", "confidence": detector.conf})
    return jsonify({"confidence": detector.conf})


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  YOLOv8 Device Detector — Web Edition")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55)
    # threaded=True so the MJPEG stream and API can be served simultaneously
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
