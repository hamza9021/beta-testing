# ─────────────────────────────────────────────
#  object_detector.py  —  Core Detection Engine
# ─────────────────────────────────────────────

import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

# Suppress verbose OpenCV/FFMPEG backend warnings on Windows
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from config import (
    MODEL_PATH,
    TARGET_CLASSES,
    CONFIDENCE_THRESHOLD,
    WEBCAM_INDEX,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
)
from utils import draw_detections, draw_fps, draw_target_legend, FPSCounter


class ObjectDetector:
    """
    Real-time object detector for Mobile Phones, Laptops, and Smart Watches.
    Uses YOLOv8 with OpenCV for webcam capture and display.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        source=None,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        """
        Args:
            model_path    : Path to .pt model file (downloads automatically if not found)
            source        : Webcam index (int) or video/image file path (str). None = default webcam.
            conf_threshold: Minimum confidence score to show a detection.
        """
        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf_threshold
        self.source = source if source is not None else WEBCAM_INDEX
        self.fps_counter = FPSCounter(window=30)

    # ──────────────────────────────────────────────────────────────────────
    def _open_capture(self) -> cv2.VideoCapture:
        """
        Open the video capture source.
        For integer (webcam) sources on Windows, auto-probes backends:
          CAP_DSHOW → CAP_MSMF → default (CAP_ANY)
        across indices 0–3 until one works.
        """
        import sys

        # ── File / URL path → open directly ───────────────────────────────
        if isinstance(self.source, str):
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(
                    f"[ERROR] Cannot open file/URL '{self.source}'."
                )
            return cap

        # ── Webcam (integer index) → probe backends on Windows ─────────────
        is_windows = sys.platform.startswith("win")

        # Ordered list of backends to try (Windows prefers DSHOW)
        if is_windows:
            backends = [
                ("CAP_DSHOW", cv2.CAP_DSHOW),
                ("CAP_MSMF",  cv2.CAP_MSMF),
                ("Default",   cv2.CAP_ANY),
            ]
        else:
            backends = [("Default", cv2.CAP_ANY)]

        # Indices to scan: requested index first, then 0-3 as fallback
        indices = [self.source] + [i for i in range(4) if i != self.source]

        MAX_ATTEMPTS = 3
        for attempt in range(1, MAX_ATTEMPTS + 1):
            for idx in indices:
                for bname, backend in backends:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Verify we can actually read a frame
                        ret, _ = cap.read()
                        if ret:
                            print(f"[INFO] Camera opened — index={idx}  backend={bname}")
                            # Reopen cleanly after the test read
                            cap.release()
                            cap = cv2.VideoCapture(idx, backend)
                            if DISPLAY_WIDTH and DISPLAY_HEIGHT:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
                            return cap
                    cap.release()

            if attempt < MAX_ATTEMPTS:
                print(f"[WARN] Camera not ready (attempt {attempt}/{MAX_ATTEMPTS}), retrying in 1s…")
                time.sleep(1.0)


        raise RuntimeError(
            "[ERROR] No webcam found on indices 0-3 with any backend.\n"
            "  • Make sure your webcam is physically connected.\n"
            "  • Close any app that may be using the camera (Teams, Zoom, Camera app).\n"
            "  • Try: python run.py --source 1"
        )

    # ──────────────────────────────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray):
        """
        Run YOLO inference on one frame and return filtered detections.

        Returns:
            boxes     : list of [x1, y1, x2, y2]
            class_ids : list of int
            scores    : list of float
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        boxes, class_ids, scores = [], [], []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                score = float(box.conf[0])
                xyxy  = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy)
                class_ids.append(cls_id)
                scores.append(score)

        return boxes, class_ids, scores

    # ──────────────────────────────────────────────────────────────────────
    def run(self):
        """Start the detection loop. Press Q or Esc to exit."""
        cap = self._open_capture()
        print("[INFO] Detection running — press  Q / Esc  to quit.")
        print(f"[INFO] Watching for: {list(TARGET_CLASSES.values())}")

        window_name = "YOLOv8 | Device Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No frame received — end of stream or disconnected camera.")
                break

            # ── Run detection ──────────────────────────────────────────────
            boxes, class_ids, scores = self._process_frame(frame)

            # ── Annotate ───────────────────────────────────────────────────
            frame = draw_detections(frame, boxes, class_ids, scores, TARGET_CLASSES)
            fps   = self.fps_counter.tick()
            frame = draw_fps(frame, fps)
            frame = draw_target_legend(frame, TARGET_CLASSES)

            # ── Detection count banner ─────────────────────────────────────
            count_text = f"Detected: {len(boxes)} object(s)"
            cv2.putText(frame, count_text, (12, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # ── Display ────────────────────────────────────────────────────
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or Esc
                print("[INFO] User requested quit.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Capture released. Goodbye!")
