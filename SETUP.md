# Setup Guide — Real-Time Collision Detection on Jetson Orin Nano

Tested hardware and software configuration that produces a working YOLO26 GPU inference pipeline with a CSI IMX219 camera.

---

## Hardware

| Component | Details |
|---|---|
| Board | NVIDIA Jetson Orin Nano |
| Camera | IMX219 CSI (connected to CAM0 port) |
| Architecture | aarch64 |

---

## System Software

| Component | Version |
|---|---|
| JetPack / L4T | R36.4.7 (JetPack 6.x) |
| Ubuntu | 22.04 |
| CUDA | 12.6 |
| cuDNN | 9.3.0.75 (`libcudnn9-cuda-12`) |
| GStreamer | 1.20.3 |
| Python | 3.10.12 |

> **Note:** The generic `torch+cu126` wheel from pytorch.org is **not compatible** with the Jetson Orin Nano (SM 8.7 compute capability). The NVIDIA Jetson-native wheel must be used instead.

---

## Python Dependencies

| Package | Version | Source |
|---|---|---|
| `torch` | `2.5.0a0+872d972e41.nv24.08` | NVIDIA Jetson JP61 wheel (see below) |
| `torchvision` | `0.20.1+cu124` | `https://download.pytorch.org/whl/cu124` |
| `numpy` | `1.26.4` | pip |
| `ultralytics` | `8.4.41` | pip |
| `Pillow` | `12.2.0` | pip |
| `opencv` | `4.8.0` (system) | system package — **not** pip |

`opencv-python` from pip must **not** be installed. It is built without GStreamer support and breaks CSI camera access. The system OpenCV at `/usr/lib/python3.10/dist-packages/cv2/` is used instead.

---

## Step-by-Step Installation

### 1. System packages

```bash
sudo apt-get install -y python3-tk python3-full
```

### 2. Create virtual environment

```bash
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install PyTorch (Jetson-native wheel)

```bash
pip install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

This wheel is compiled specifically for SM 8.7 (Orin) and links against the system CUDA/cuDNN. Generic pytorch.org wheels fail at runtime with `GET was unable to find an engine to execute this computation`.

### 4. Install torchvision (no-deps to prevent torch overwrite)

```bash
pip install torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --no-deps
```

`--no-deps` is required. Without it, pip replaces the Jetson torch wheel with the incompatible `torch==2.5.1` from pytorch.org.

### 5. Install remaining packages

```bash
pip install "ultralytics>=8.4.41" "numpy>=1.23.0,<2.0.0" "Pillow>=10.0.0"
```

Do **not** include `opencv-python` here.

### 6. Install libcusparseLt

`libcusparseLt.so.0` is absent from the Jetson system CUDA installation but is required by the torch wheel:

```bash
pip install nvidia-cusparselt-cu12 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 7. Expose system OpenCV to the venv

```bash
echo "/usr/lib/python3.10/dist-packages" \
  > venv/lib/python3.10/site-packages/system_cv2.pth
```

This makes the system `cv2` (4.8.0, with GStreamer support) importable from within the venv.

---

## Required Patches

Two torchvision files must be patched because `torchvision 0.20.1` was compiled against `torch 2.5.1` while the Jetson wheel is `torch 2.5.0a0`, causing ABI mismatches at two points.

### Patch 1 — `_meta_registrations.py`

The Jetson torch registers some torchvision ops as `CompositeImplicitAutograd`, which causes a `RuntimeError` when torchvision tries to add meta kernels for them at import time.

File: `venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py`

Find:
```python
        if torchvision.extension._has_ops():
            get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)
```

Replace with:
```python
        if torchvision.extension._has_ops():
            try:
                get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)
            except RuntimeError:
                pass
```

### Patch 2 — `ops/boxes.py`

The CPU/CUDA dispatch keys are swapped between `torch 2.5.0a0` and `2.5.1`, making `torch.ops.torchvision.nms` fail on both CPU and CUDA tensors. Replace it with a pure-PyTorch NMS that needs no C++ extension.

File: `venv/lib/python3.10/site-packages/torchvision/ops/boxes.py`

Add the following function **above** the `nms()` function:

```python
def _pure_torch_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        inter_w = (torch.minimum(x2[i], x2[rest]) - torch.maximum(x1[i], x1[rest])).clamp(min=0)
        inter_h = (torch.minimum(y2[i], y2[rest]) - torch.maximum(y1[i], y1[rest])).clamp(min=0)
        iou = (inter_w * inter_h) / (areas[i] + areas[rest] - inter_w * inter_h)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)
```

Then change the last line of the `nms()` function from:
```python
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)
```
to:
```python
    return _pure_torch_nms(boxes, scores, iou_threshold)
```

> Both patches are applied automatically by `setup.sh`.

---

## Configuration

### Passwordless nvargus-daemon restart

`run.sh` restarts the CSI camera daemon on every launch to prevent stale sessions from a previous crash. Run this once to allow it without a password prompt:

```bash
echo "%sudo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nvargus-daemon" \
  | sudo tee /etc/sudoers.d/nvargus-daemon
sudo chmod 0440 /etc/sudoers.d/nvargus-daemon
```

---

## Automated Setup

All of the above is handled by `setup.sh`. Run it once after cloning:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

Then apply the sudoers rule (one-time, requires your password):

```bash
echo "%sudo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nvargus-daemon" \
  | sudo tee /etc/sudoers.d/nvargus-daemon && sudo chmod 0440 /etc/sudoers.d/nvargus-daemon
```

---

## Running the Program

```bash
# YOLO26s with CSI camera (recommended)
./run.sh yolo --csi

# YOLO26s with USB webcam
./run.sh yolo

# If using a second CSI camera port
./run.sh yolo --csi --sensor-id 1
```

`run.sh` automatically selects the GPU version when CUDA is available and falls back to CPU otherwise.

---

## Known Issues and Notes

### `LD_LIBRARY_PATH` for cusparseLt

`run.sh` exports the cusparseLt lib path before launching Python:

```bash
export LD_LIBRARY_PATH="<venv>/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
```

Only this one directory is added. Adding all pip nvidia lib directories causes the pip-bundled cuDNN 9.1 to override the system cuDNN 9.3, which breaks GPU execution (`CUBLAS_STATUS_ALLOC_FAILED`).

### First launch delay

On the very first run, PyTorch JIT-compiles CUDA kernels for the Orin's SM 8.7. This adds ~30–90 seconds before the window opens. Subsequent launches are instant.

### CSI camera "Failed to create CaptureSession"

Caused by a stale argus session left over from a previous crash. The `run.sh` daemon restart handles this automatically. If running a script directly (bypassing `run.sh`):

```bash
sudo systemctl restart nvargus-daemon
```

### `Cannot query video position` GStreamer warning

Harmless. GStreamer cannot seek in a live camera stream. Does not affect frame capture.

### `NvMapMemAllocInternalTagged: error 12`

Occasional memory pressure warnings from the camera subsystem. Usually harmless for the YOLO26s model. If they cause crashes with larger models, reduce the inference resolution (`imgsz` in the predict call).
