#!/usr/bin/env python
# ─────────────────────────────────────────────
#  run.py  —  Entry Point
#
#  Usage:
#    python run.py                          # default webcam
#    python run.py --source 1              # second webcam
#    python run.py --source video.mp4      # video file
#    python run.py --conf 0.55             # higher confidence threshold
#    python run.py --model yolov8s.pt      # larger/more accurate model
# ─────────────────────────────────────────────

import argparse
import sys

from object_detector import ObjectDetector
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, WEBCAM_INDEX


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Object Detector — Mobile Phone / Laptop / Smart Watch",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Video source:\n"
            "  0, 1, 2 …  webcam index  (default: 0)\n"
            "  path/to/video.mp4         video file"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold 0–1  (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=(
            f"YOLOv8 model file  (default: {MODEL_PATH})\n"
            "Options: yolov8n.pt  yolov8s.pt  yolov8m.pt  yolov8l.pt  yolov8x.pt"
        ),
    )
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  YOLOv8 Device Detector — Mobile / Laptop / Watch")
    print("=" * 55)

    args = parse_args()

    # Resolve --source: integer webcam index or string file path
    source = args.source
    if source is not None:
        try:
            source = int(source)          # webcam index
        except ValueError:
            pass                          # keep as file path string

    try:
        detector = ObjectDetector(
            model_path=args.model,
            source=source,
            conf_threshold=args.conf,
        )
        detector.run()
    except RuntimeError as err:
        print(err)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
