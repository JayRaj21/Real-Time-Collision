# Real-Time Collision Detection

Live object detection with bounding-box collision highlighting, a side-panel metrics display, and support for both USB webcams and CSI cameras (IMX219). Two model families are available: YOLO26s (fast) and RT-DETR-L (higher accuracy).

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| **Runtime** | Python | 3.10.12 |
| **ML framework** | PyTorch (NVIDIA Jetson wheel) | 2.5.0a0+nv24.08 |
| **Vision library** | torchvision | 0.20.1 |
| **Detection models** | Ultralytics (YOLO26s, RT-DETR-L) | 8.4.41 |
| **Computer vision** | OpenCV (system build, GStreamer-enabled) | 4.8.0 |
| **Camera interface** | GStreamer + nvarguscamerasrc | 1.20.3 |
| **GUI** | tkinter | system |
| **Numerical** | NumPy | 1.26.4 |
| **Image processing** | Pillow | 12.2.0 |
| **Accelerator** | CUDA | 12.6 |
| **cuDNN** | NVIDIA cuDNN | 9.3.0 |
| **Platform** | NVIDIA Jetson Orin Nano (SM 8.7) | JetPack R36.4.7 |

---

## Features

- Real-time object detection at up to 59 FPS on the Jetson Orin Nano GPU
- Choice of YOLO26s (fast) or RT-DETR-L (transformer, higher accuracy)
- CSI camera support (IMX219 via `nvarguscamerasrc`) alongside USB webcam
- Side panel showing live FPS, latency breakdown, detection counts, confidence scores, and rolling quality estimates
- Optional bounding-box collision detection with visual highlights
- Automatic GPU/CPU fallback via `run.sh`

---

## Hardware

Developed and tested on:

- **Board:** NVIDIA Jetson Orin Nano
- **Camera:** IMX219 CSI (connected to CAM0 / sensor 0)
- **OS:** Ubuntu 22.04 (JetPack R36.4.7)
- **CUDA:** 12.6 · **cuDNN:** 9.3.0 · **GPU:** Orin (SM 8.7)

---

## Quick Start

```bash
git clone <repo-url>
cd Real-Time-Collision-3
chmod +x setup.sh run.sh
./setup.sh
```

Then add the one-time passwordless camera-daemon rule (enter your password when prompted):

```bash
echo "%sudo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nvargus-daemon" \
  | sudo tee /etc/sudoers.d/nvargus-daemon && \
  sudo chmod 0440 /etc/sudoers.d/nvargus-daemon
```

Run:

```bash
./run.sh yolo --csi      # YOLO26s + IMX219 CSI camera
./run.sh yolo            # YOLO26s + USB webcam
./run.sh rtdetr --csi    # RT-DETR-L + IMX219 CSI camera
```

---

## Full Setup (what setup.sh does)

### 1. System packages

```bash
sudo apt-get install -y python3-tk python3-full
```

### 2. Virtual environment

```bash
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. PyTorch — Jetson-native wheel

The standard pytorch.org wheel does not include a native CUDA kernel for Jetson Orin (SM 8.7) and will fail at runtime. Install the NVIDIA Jetson JP61 build instead:

```bash
pip install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### 4. torchvision

Must be installed with `--no-deps` to prevent pip from replacing the Jetson torch wheel with an incompatible version:

```bash
pip install torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --no-deps
```

### 5. Remaining packages

```bash
pip install "ultralytics>=8.4.41" "numpy>=1.23.0,<2.0.0" "Pillow>=10.0.0"
pip install nvidia-cusparselt-cu12 \
  --index-url https://download.pytorch.org/whl/cu126
```

Do **not** install `opencv-python`. The system OpenCV (4.8.0 with GStreamer) is used for CSI camera access.

### 6. Expose system OpenCV to the venv

```bash
echo "/usr/lib/python3.10/dist-packages" \
  > venv/lib/python3.10/site-packages/system_cv2.pth
```

### 7. Apply torchvision compatibility patches

`setup.sh` applies two patches automatically. They fix ABI mismatches between `torchvision 0.20.1` (compiled for `torch 2.5.1`) and the Jetson wheel (`torch 2.5.0a0`):

- **`_meta_registrations.py`** — silences meta-kernel registration errors that occur at import time
- **`ops/boxes.py`** — replaces the broken C++ NMS dispatch with a pure-PyTorch implementation

Both patches are re-applied whenever `setup.sh` is run.

### 8. Passwordless daemon restart (one-time, requires sudo)

`run.sh` restarts `nvargus-daemon` on every launch to clear stale CSI sessions. Allow this without a password prompt:

```bash
echo "%sudo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nvargus-daemon" \
  | sudo tee /etc/sudoers.d/nvargus-daemon && \
  sudo chmod 0440 /etc/sudoers.d/nvargus-daemon
```

---

## Running

### Commands

```bash
./run.sh yolo              # YOLO26s, USB webcam
./run.sh yolo --csi        # YOLO26s, IMX219 CSI camera
./run.sh rtdetr            # RT-DETR-L, USB webcam
./run.sh rtdetr --csi      # RT-DETR-L, IMX219 CSI camera
./run.sh yolo --csi --sensor-id 1   # second CSI port
```

`run.sh` automatically picks the GPU variant when CUDA is available and falls back to CPU otherwise.

### Available programs

| Script | Model | Device | Notes |
|---|---|---|---|
| `detect_gpu.py` | YOLO26s | CUDA | Recommended |
| `detect_cpu.py` | YOLO26s | CPU | Fallback |
| `detect_rtdetr_gpu.py` | RT-DETR-L | CUDA | Higher accuracy, slower |
| `detect_rtdetr_cpu.py` | RT-DETR-L | CPU | Very slow (~1–3 FPS) |

### Expected performance (Jetson Orin Nano, CSI camera, 1280×720)

| Model | Device | Typical FPS |
|---|---|---|
| YOLO26s | GPU (CUDA) | 30–59 FPS |
| YOLO26s | CPU | 4–10 FPS |
| RT-DETR-L | GPU (CUDA) | 8–15 FPS |
| RT-DETR-L | CPU | 1–3 FPS |

---

## Troubleshooting

**`ImportError: libcusparseLt.so.0`**
Run via `run.sh` — it sets the required `LD_LIBRARY_PATH`. Do not invoke `python3 detect_gpu.py` directly without it.

**`GET was unable to find an engine to execute this computation`**
Wrong PyTorch wheel installed. Re-run `setup.sh` to reinstall the Jetson-native wheel.

**`Error: Failed to create CaptureSession`**
Stale argus session from a previous crash. `run.sh` handles this automatically. To fix manually:
```bash
sudo systemctl restart nvargus-daemon
```

**Black screen / V4L2 timeout on `/dev/video0`**
The IMX219 CSI camera does not respond to V4L2. Always pass `--csi` when using it.

**`torchvision::nms` errors (various)**
The torchvision patches are missing. Re-run `setup.sh`.

**`NvMapMemAllocInternalTagged: error 12` with RT-DETR-L**
Unified memory pressure. RT-DETR-L is large — reduce the inference resolution or switch to YOLO26s.

**`GStreamer warning: Cannot query video position`**
Harmless. GStreamer cannot seek on a live stream.
