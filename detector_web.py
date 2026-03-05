# ─────────────────────────────────────────────
#  detector_web.py  —  Headless Detection Engine for Flask
# ─────────────────────────────────────────────

import cv2
import time
import os
import threading
import numpy as np
from ultralytics import YOLO

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from config import (
    MODEL_PATH,
    TARGET_CLASSES,
    CLASS_COLORS,
    CONFIDENCE_THRESHOLD,
    WEBCAM_INDEX,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    FONT_SCALE,
    FONT_THICKNESS,
    BOX_THICKNESS,
)


class WebDetector:
    """
    Background-thread detector that continually grabs webcam frames,
    runs YOLOv8 inference, annotates them, and stores the latest JPEG
    so the Flask route can serve it as an MJPEG stream.
    """

    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = CONFIDENCE_THRESHOLD):
        print(f"[INFO] Loading model: {model_path}")
        self.model          = YOLO(model_path)
        self.conf           = conf_threshold
        self._lock          = threading.Lock()
        self._frame_bytes   = b""
        self._stats         = {"fps": 0.0, "detections": {}, "total": 0}
        self._running       = False
        self._thread        = None
        self._cap           = None

        # Rolling FPS
        self._times = []
        self._fps_window = 30

    # ── public API ──────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[INFO] Detector thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        print("[INFO] Detector stopped.")

    def get_frame(self) -> bytes:
        with self._lock:
            return self._frame_bytes

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def set_confidence(self, val: float):
        self.conf = max(0.01, min(1.0, val))

    # ── internal ────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        import sys
        is_windows = sys.platform.startswith("win")
        backends = (
            [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF), ("Default", cv2.CAP_ANY)]
            if is_windows else [("Default", cv2.CAP_ANY)]
        )
        indices = [WEBCAM_INDEX] + [i for i in range(4) if i != WEBCAM_INDEX]

        for attempt in range(1, 4):
            for idx in indices:
                for bname, backend in backends:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            print(f"[INFO] Camera opened — index={idx} backend={bname}")
                            cap.release()
                            cap = cv2.VideoCapture(idx, backend)
                            if DISPLAY_WIDTH and DISPLAY_HEIGHT:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
                            return cap
                    cap.release()
            if attempt < 3:
                print(f"[WARN] Camera not ready (attempt {attempt}/3), retrying in 1s…")
                time.sleep(1.0)

        raise RuntimeError(
            "[ERROR] No webcam found. Make sure it is connected and not used by another app."
        )

    def _tick_fps(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._fps_window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    def _draw(self, frame: np.ndarray, boxes, class_ids, scores) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX

        for box, cls_id, score in zip(boxes, class_ids, scores):
            x1, y1, x2, y2 = map(int, box)
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label = TARGET_CLASSES.get(cls_id, f"Class {cls_id}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            caption = f"{label}  {score:.0%}"
            (tw, th), baseline = cv2.getTextSize(caption, font, FONT_SCALE, FONT_THICKNESS)
            bg_y1 = max(y1 - th - baseline - 8, 0)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, cv2.FILLED)
            cv2.putText(frame, caption, (x1 + 4, y1 - baseline - 2),
                        font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

        # FPS overlay
        fps = self._tick_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 36),
                    font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Detection count
        cv2.putText(frame, f"Detected: {len(boxes)} object(s)", (12, 72),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Legend (bottom-left)
        h = frame.shape[0]
        y_start = h - (len(TARGET_CLASSES) * 28) - 12
        for i, (cls_id, name) in enumerate(TARGET_CLASSES.items()):
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            y = y_start + i * 28
            cv2.rectangle(frame, (10, y), (24, y + 18), color, cv2.FILLED)
            cv2.putText(frame, name, (30, y + 14),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame, fps

    def _loop(self):
        self._cap = self._open_capture()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                print("[WARN] No frame received — end of stream.")
                break

            # ── Inference ──────────────────────────────────────────────────
            results = self.model(frame, conf=self.conf, verbose=False)[0]
            boxes, class_ids, scores = [], [], []

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in TARGET_CLASSES:
                        continue
                    scores.append(float(box.conf[0]))
                    boxes.append(box.xyxy[0].cpu().numpy())
                    class_ids.append(cls_id)

            # ── Annotate ───────────────────────────────────────────────────
            frame, fps = self._draw(frame, boxes, class_ids, scores)

            # ── Build stats dict ───────────────────────────────────────────
            det_counts = {}
            for cls_id in class_ids:
                name = TARGET_CLASSES[cls_id]
                det_counts[name] = det_counts.get(name, 0) + 1

            # ── Encode to JPEG ─────────────────────────────────────────────
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._frame_bytes = jpeg.tobytes()
                    self._stats = {
                        "fps": round(fps, 1),
                        "detections": det_counts,
                        "total": len(boxes),
                    }

        if self._cap:
            self._cap.release()
