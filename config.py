# ─────────────────────────────────────────────
#  config.py  —  Detection Settings
# ─────────────────────────────────────────────

# YOLOv8 model file (nano = fastest; swap for yolov8s/m/l/x for accuracy)
MODEL_PATH = "yolov8n.pt"

# COCO class IDs for our three target devices
#   67 → cell phone
#   63 → laptop  (COCO uses 63 for laptop)
#   49 → clock   (closest match to smartwatch in COCO)
TARGET_CLASSES = {
    67: "Mobile Phone",
    63: "Laptop / Notebook",
    49: "Smart Watch",
}

# Minimum confidence to display a detection (0–1)
CONFIDENCE_THRESHOLD = 0.40

# Webcam index (0 = default camera)
WEBCAM_INDEX = 0

# Display resolution (width, height)  —  0,0 = use native camera resolution
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# Bounding-box colours per class (BGR format for OpenCV)
CLASS_COLORS = {
    67: (0,  220, 255),   # Yellow-ish  → Mobile Phone
    63: (0,  255, 120),   # Green       → Laptop
    49: (255, 80,  80),   # Blue-ish    → Smart Watch
}

# Overlay font settings
FONT            = "FONT_HERSHEY_SIMPLEX"
FONT_SCALE      = 0.75
FONT_THICKNESS  = 2
BOX_THICKNESS   = 2
