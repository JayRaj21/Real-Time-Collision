# Real-Time Object Detection with YOLO26

Live webcam object detection using [YOLO26n](https://docs.ultralytics.com/models/yolo26/) — the nano variant of Ultralytics' latest model. Bounding boxes, class labels, confidence scores, and a live FPS counter are overlaid on the video feed.

---

## Requirements

- Python 3.9+
- A connected webcam (uses `/dev/video0` by default)
- Linux (tested), should also work on macOS/Windows with minor path adjustments

---

## Installation

### 1. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install CPU dependencies

```bash
pip install -r requirements.txt
```

### 3. (GPU only) Install PyTorch with CUDA support

The `requirements.txt` installs the CPU-only build of PyTorch by default. If you have an NVIDIA GPU, replace it with the CUDA-enabled build **before** installing the rest:

```bash
# Example for CUDA 12.1 — adjust the index URL for your CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python
```

The model weights (`yolo26n.pt`) are downloaded automatically on first run.

---

## Running

### GPU version (NVIDIA CUDA)

```bash
python detect_gpu.py
```

On startup the script verifies that CUDA is available. If no compatible GPU is found it prints an error and tells you to run the CPU version instead.

### CPU version

```bash
python detect_cpu.py
```

No GPU required. Works on any machine with Python and a webcam.

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit and close the window |

---

## Expected FPS

| Hardware | Typical FPS |
|----------|------------|
| Modern NVIDIA GPU (RTX 30/40 series) | 60 – 200+ FPS |
| Older / mid-range GPU (GTX 10/16 series) | 30 – 80 FPS |
| Modern CPU (8-core, e.g. Ryzen 5 / Core i5) | 8 – 20 FPS |
| Older / low-power CPU | 2 – 8 FPS |

YOLO26n is the nano variant — it is the fastest and most memory-efficient option. For higher accuracy at the cost of speed, swap `yolo26n.pt` for `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, or `yolo26x.pt` in either script.

---

## Troubleshooting

**Webcam not found**
- Check that no other application is using the camera.
- Try a different device index: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.

**`libGL` error on headless / minimal Linux**
```bash
sudo apt-get install libgl1-mesa-glx
```

**Low FPS on CPU**
- Lower the input resolution by adding `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` after `VideoCapture(0)`.
- Raise `CONF_THRESHOLD` (e.g. to `0.5`) to skip low-confidence detections faster.
