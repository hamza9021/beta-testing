# ─────────────────────────────────────────────
#  utils.py  —  Drawing & FPS Helpers
# ─────────────────────────────────────────────

import time
import cv2
import numpy as np
from config import CLASS_COLORS, FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS


def draw_detections(frame: np.ndarray, boxes, class_ids, scores, class_names: dict) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.

    Args:
        frame      : BGR image (numpy array)
        boxes      : list of [x1, y1, x2, y2] in pixel coordinates
        class_ids  : list of COCO class IDs
        scores     : list of confidence scores (0–1)
        class_names: dict mapping class_id -> human-readable name

    Returns:
        Annotated frame (same array, modified in place)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, cls_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = class_names.get(cls_id, f"Class {cls_id}")

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # ── Filled label background ───────────────────────────────────────
        caption = f"{label}  {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(caption, font, FONT_SCALE, FONT_THICKNESS)
        bg_y1 = max(y1 - th - baseline - 8, 0)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, cv2.FILLED)

        # ── Label text ────────────────────────────────────────────────────
        text_color = (0, 0, 0)   # black on coloured background
        cv2.putText(frame, caption, (x1 + 4, y1 - baseline - 2),
                    font, FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (12, 36), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_target_legend(frame: np.ndarray, class_names: dict) -> np.ndarray:
    """Draw a small legend in the bottom-left corner showing detected classes."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = frame.shape[0]
    y_start = h - (len(class_names) * 28) - 12

    for idx, (cls_id, name) in enumerate(class_names.items()):
        from config import CLASS_COLORS
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        y = y_start + idx * 28
        cv2.rectangle(frame, (10, y), (24, y + 18), color, cv2.FILLED)
        cv2.putText(frame, name, (30, y + 14), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


class FPSCounter:
    """Simple rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
