# 🎯 YOLOv8 Device Detector — Mobile / Laptop / Smart Watch

Real-time object detection using **YOLOv8** + **OpenCV** for three device categories:

| COCO Class ID | Label | Device |
|---|---|---|
| 67 | `cell phone` | 📱 Mobile Phone |
| 63 | `laptop` | 💻 Laptop / Notebook |
| 49 | `clock` | ⌚ Smart Watch |

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

> On first run, `yolov8n.pt` (~6 MB) downloads automatically from the Ultralytics CDN.

---

## ▶️ Running

```bash
# Default webcam (index 0)
python run.py

# Second webcam
python run.py --source 1

# Video file
python run.py --source path/to/video.mp4

# Higher accuracy model
python run.py --model yolov8s.pt

# Adjust confidence threshold (0.0 – 1.0)
python run.py --conf 0.55
```

Press **Q** or **Esc** to quit.

---

## 📁 Project Structure

```
PROJECT BETA TESTING/
├── run.py              ← Entry point (start here)
├── object_detector.py  ← Core detection engine
├── config.py           ← Settings (classes, colors, thresholds)
├── utils.py            ← Drawing & FPS helpers
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 🖥️ On-Screen Overlay

- **Colored bounding boxes** per device type
- **Label + confidence %** above each box
- **FPS counter** — top left
- **Detection count** — below FPS
- **Class legend** — bottom left

---

## 🔧 Customization

| What | Where | How |
|---|---|---|
| Add/remove classes | `config.py` → `TARGET_CLASSES` | Add COCO class IDs |
| Change colors | `config.py` → `CLASS_COLORS` | BGR tuple per class |
| Switch model | `--model yolov8s.pt` or edit `config.py` | n/s/m/l/x variants |
| Confidence | `--conf 0.5` or edit `config.py` | 0.0 – 1.0 |
