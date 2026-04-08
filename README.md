# Jetson Collision Detector

Real-time object detection and collision/contact event identification running
on a **Jetson Orin Nano** with a **Waveshare IMX219-160 CSI camera**.

- Detects objects with **YOLOv8n** running on the Jetson GPU (CUDA)
- Identifies when any two detected bounding boxes touch or overlap
- Draws a red line between colliding objects and labels the event (e.g. `book ↔ table`)
- Tkinter/X11 GUI with a live camera feed and a "Show Boxes" toggle

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| JetPack | 5.1.2 (L4T r35) or 6.0 (L4T r36) |
| Python | 3.8+ (3.10 on JetPack 6) |
| CUDA | 11.4+ (bundled with JetPack) |

---

## 1 — Enable the CSI camera

### 1a. Physical connection
Seat the IMX219-160 ribbon cable into the **CAM0** connector on the Orin Nano
carrier board with the blue strip facing toward the board edge.

### 1b. Verify the camera is visible to the driver
```bash
# Should print at least one /dev/video* device
ls /dev/video*

# Quick GStreamer smoke-test (captures one frame and discards it)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink
```

If `nvarguscamerasrc` fails with *"No cameras available"*, check:
- The ribbon cable is seated in CAM0 (not CAM1)
- `sudo systemctl status nvargus-daemon` — the Argus daemon must be running
- The device tree overlay for the IMX219 is enabled

### 1c. Enable the camera overlay (if not already active)
```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
# Navigate to: Configure Jetson 24pin CSI Connector
# Enable: Camera IMX219 Dual
# Save and reboot
```

---

## 2 — Install JetPack-compatible PyTorch

> **Do not** run `pip install torch` — the PyPI wheel is x86-64 only and will
> either fail or run on CPU without CUDA.

### JetPack 6 (L4T r36, CUDA 12.x) — Python 3.10
```bash
# Download the NVIDIA-provided aarch64 wheel
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl

pip install torch-2.3.0a0+ebedce2.nv24.02-cp310-cp310-linux_aarch64.whl
```

### JetPack 5.x (L4T r35, CUDA 11.4) — Python 3.8
```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

Find the full list of available wheels at:
<https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048>

### Verify CUDA is accessible
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Expected output: True
```

---

## 3 — Install GStreamer dependencies

JetPack ships the required GStreamer Jetson plugins. Confirm and install any
missing pieces:

```bash
sudo apt update
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# Jetson-specific GStreamer elements (nvarguscamerasrc, nvvidconv, …)
# These are part of nvidia-l4t-gstreamer — installed by JetPack automatically.
# If missing:
sudo apt install -y nvidia-l4t-gstreamer
```

### GStreamer-enabled OpenCV

The app requires the system `python3-opencv` package, which is compiled with
GStreamer support. **Do not** replace it with the PyPI `opencv-python` wheel.

```bash
sudo apt install -y python3-opencv

# Verify GStreamer support is present
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
# Should show:  GStreamer:   YES
```

---

## 4 — Install Python dependencies

```bash
cd jetson-collision-detector

# Install remaining packages (ultralytics, Pillow, numpy)
pip install -r requirements.txt
```

`yolov8n.pt` (~6 MB) is downloaded automatically on the first run.

---

## 5 — Run the application

### Running at the physical desktop
```bash
python3 main.py
```

### Running over SSH with X forwarding
```bash
# On your laptop/desktop:
ssh -X jetson@<jetson-ip>

# On the Jetson:
export DISPLAY=:0   # use the Jetson's own display, or :10 for X forwarding
python3 main.py
```

If `DISPLAY` is not set, the app sets it to `:0` automatically.

### Expected startup output
```
[init] Opening CSI camera ...
[init] Camera opened OK  (1280×720)
[init] Loading YOLOv8n on device=CUDA ...
[init] Model ready.
[init] GUI started.  Close the window or press Ctrl-C to quit.
```

---

## 6 — GUI controls

| Control | Function |
|---|---|
| **Show Boxes** checkbox | Toggle all per-object bounding boxes on/off |
| **FPS** badge (top-right) | Rolling 1-second frame rate of the display loop |
| **Contacts** badge | Number of active collision/contact events in the current frame |
| Window close button / Ctrl-C | Graceful shutdown |

Collision events are always visible regardless of the "Show Boxes" setting:
a red line connects the two objects' centres, and a label such as
`person ↔ chair` appears at the contact midpoint.

---

## 7 — Performance notes

| Parameter | Value |
|---|---|
| YOLO input size | 640 × 640 |
| Camera resolution | 1280 × 720 @ 30 fps |
| Display resolution | 960 × 540 |
| Target FPS | ≥ 20 |

Inference runs in a dedicated thread so the display loop is never stalled
waiting for YOLO results. On a Jetson Orin Nano 8 GB you should see
25–30 FPS in typical indoor scenes.

To increase throughput further:
- Lower `YOLO_SIZE` to 320 in `main.py` (faster, slightly less accurate)
- Lower `CONF_THRESH` to 0.30 or raise it to 0.50 to tune detections
- Use `yolov8n.engine` (TensorRT export) instead of `yolov8n.pt`:
  ```bash
  yolo export model=yolov8n.pt format=engine device=0
  ```
  Then change `YOLO("yolov8n.pt")` to `YOLO("yolov8n.engine")` in `main.py`.

---

## 8 — Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `RuntimeError: Cannot open CSI camera` | Argus daemon not running, or ribbon cable issue — run the GStreamer smoke-test in §1b |
| `torch.cuda.is_available()` returns `False` | PyPI torch installed instead of Jetson wheel — reinstall per §2 |
| `ImportError: No module named 'cv2'` | `python3-opencv` not installed — run `sudo apt install python3-opencv` |
| `_tkinter.TclError: no display` | SSH session without X forwarding — set `DISPLAY=:0` to use the Jetson's local display |
| Low FPS (< 15) | Thermal throttling — check `sudo tegrastats`; ensure the Jetson has adequate cooling |
| Black canvas, no image | Camera opens but reads no frames — check `nvargus-daemon` and ribbon cable seating |
