# ─────────────────────────────────────────────
#  detector_web.py  —  Stateless Inference Engine (no camera on server)
# ─────────────────────────────────────────────
#
#  The server never opens a webcam.
#  Instead, call process_frame(jpeg_bytes) with raw JPEG bytes sent
#  from the browser, and get back (annotated_jpeg_bytes, stats_dict).
# ─────────────────────────────────────────────

import cv2
import time
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    TARGET_CLASSES,
    CLASS_COLORS,
    CONFIDENCE_THRESHOLD,
    FONT_SCALE,
    FONT_THICKNESS,
    BOX_THICKNESS,
)


class WebDetector:
    """
    Stateless YOLO inference helper.
    No threads, no camera — just pure frame → annotated-JPEG processing.
    """

    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = CONFIDENCE_THRESHOLD):
        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf_threshold

    # ── public API ────────────────────────────────────────────────────────────

    def set_confidence(self, val: float):
        self.conf = max(0.01, min(1.0, val))

    def process_frame(self, jpeg_bytes: bytes) -> tuple[bytes, dict]:
        """
        Accept raw JPEG bytes from the browser.
        Returns (annotated_jpeg_bytes, stats_dict).
        """
        # Decode JPEG → numpy BGR
        arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return b"", {"fps": 0, "detections": {}, "total": 0}

        t0 = time.perf_counter()

        # ── Inference ─────────────────────────────────────────────────────────
        results   = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes, class_ids, scores = [], [], []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                scores.append(float(box.conf[0]))
                boxes.append(box.xyxy[0].cpu().numpy())
                class_ids.append(cls_id)

        # ── Annotate ──────────────────────────────────────────────────────────
        frame = self._draw(frame, boxes, class_ids, scores)

        elapsed = time.perf_counter() - t0
        fps     = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0

        # ── Encode back to JPEG ───────────────────────────────────────────────
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out_bytes = jpeg.tobytes() if ok else b""

        det_counts: dict = {}
        for cls_id in class_ids:
            name = TARGET_CLASSES[cls_id]
            det_counts[name] = det_counts.get(name, 0) + 1

        stats = {
            "fps":        fps,
            "detections": det_counts,
            "total":      len(boxes),
        }
        return out_bytes, stats

    # ── internal helpers ──────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, boxes, class_ids, scores) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX

        for box, cls_id, score in zip(boxes, class_ids, scores):
            x1, y1, x2, y2 = map(int, box)
            color  = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label  = TARGET_CLASSES.get(cls_id, f"Class {cls_id}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            caption = f"{label}  {score:.0%}"
            (tw, th), baseline = cv2.getTextSize(caption, font, FONT_SCALE, FONT_THICKNESS)
            bg_y1 = max(y1 - th - baseline - 8, 0)
            cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, cv2.FILLED)
            cv2.putText(frame, caption, (x1 + 4, y1 - baseline - 2),
                        font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

        # Detection count overlay
        cv2.putText(frame, f"Detected: {len(boxes)} object(s)", (12, 36),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
