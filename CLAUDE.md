# Real-Time Collision Detection — Claude Context

## What This Project Does

Tkinter GUI application that runs real-time object detection and bounding-box collision detection on a video feed. Four variants:

| Script | Model | Device |
|---|---|---|
| `detect_gpu.py` | YOLO26s | CUDA |
| `detect_cpu.py` | YOLO26s | CPU |
| `detect_rtdetr_gpu.py` | RT-DETR-L | CUDA |
| `detect_rtdetr_cpu.py` | RT-DETR-L | CPU |

Entry point is `run.sh`, which selects GPU/CPU automatically and passes arguments through.

---

## Hardware

- **Board:** NVIDIA Jetson Orin Nano — aarch64, unified CPU/GPU memory
- **GPU:** Orin (nvgpu), compute capability **SM 8.7**
- **Camera:** IMX219 CSI (sensor 0, CAM0 port)
- **JetPack:** R36.4.7 (JetPack 6.x)
- **CUDA:** 12.6 (system, at `/usr/local/cuda`)
- **cuDNN:** 9.3.0 (system, at `/usr/lib/aarch64-linux-gnu/`)
- **OS:** Ubuntu 22.04, Python 3.10.12

---

## Setting Up the Environment

When a user asks to install dependencies, set up the environment, or run the program for the first time, follow these steps in order. Most steps are automated by `setup.sh`; the sudoers rule is the only step that requires interactive user input.

### Step 1 — Check what is already done

```bash
# Check if venv exists
ls venv/bin/activate 2>/dev/null && echo "venv OK" || echo "venv MISSING"

# Check torch (requires LD_LIBRARY_PATH)
CUSPARSELT="$(pwd)/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib"
LD_LIBRARY_PATH="$CUSPARSELT" venv/bin/python3 -c "
import warnings; warnings.filterwarnings('ignore')
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
" 2>&1

# Check patches are applied
grep -q "_pure_torch_nms" venv/lib/python3.10/site-packages/torchvision/ops/boxes.py \
  && echo "Patch 2 OK" || echo "Patch 2 MISSING"
grep -q "except RuntimeError" venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py \
  && echo "Patch 1 OK" || echo "Patch 1 MISSING"

# Check system_cv2.pth
ls venv/lib/python3.10/site-packages/system_cv2.pth 2>/dev/null \
  && echo "cv2 pth OK" || echo "cv2 pth MISSING"

# Check sudoers rule
sudo -n systemctl restart nvargus-daemon 2>/dev/null \
  && echo "sudoers OK" || echo "sudoers rule MISSING — user must add it"
```

### Step 2 — Run setup.sh (handles everything except sudoers)

If the venv is missing or packages are wrong, run:

```bash
cd /home/jet/Real-Time-Collision-3
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` takes 5–15 minutes on first run (torch wheel is ~500 MB). It is safe to re-run — it recreates the venv from scratch and reapplies all patches.

### Step 3 — Add the sudoers rule (requires user's password — one time only)

`setup.sh` attempts to add this automatically, but it may fail if sudo is not configured for non-interactive use. Check first:

```bash
sudo -n systemctl restart nvargus-daemon 2>/dev/null && echo "already set" || echo "needs adding"
```

If it needs adding, ask the user to run this single command in their terminal (type it with `!` prefix to run it here):

```
! echo "%sudo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nvargus-daemon" | sudo tee /etc/sudoers.d/nvargus-daemon && sudo chmod 0440 /etc/sudoers.d/nvargus-daemon
```

### Step 4 — Verify the full stack

```bash
cd /home/jet/Real-Time-Collision-3
CUSPARSELT="$(pwd)/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib"
LD_LIBRARY_PATH="$CUSPARSELT" venv/bin/python3 -c "
import warnings; warnings.filterwarnings('ignore')
import torch, torchvision, cv2, numpy
from ultralytics import YOLO
print('torch      :', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cv2        :', cv2.__version__)
print('numpy      :', numpy.__version__)
gst = 'YES' if 'GStreamer:                   YES' in cv2.getBuildInformation() else 'NO'
print('GStreamer  :', gst)
print('CUDA       :', torch.cuda.is_available())
t = torch.randn(2, 2).cuda()
print('CUDA tensor:', t.shape, 'on', t.device)
print('All OK')
" 2>&1 | grep -v "^GST_ARGUS\|^CONSUMER\|UserWarning\|_warn_unsupported"
```

Expected output:
```
torch      : 2.5.0a0+872d972e41.nv24.08
torchvision: 0.20.1
cv2        : 4.8.0
numpy      : 1.26.4
GStreamer  : YES
CUDA       : True
CUDA tensor: torch.Size([2, 2]) on cuda:0
All OK
```

### Step 5 — Run the program

```bash
./run.sh yolo --csi
```

---

## Running the Program

```bash
# Always use run.sh — it sets LD_LIBRARY_PATH and restarts nvargus-daemon
./run.sh yolo --csi        # YOLO26s + IMX219 CSI camera
./run.sh yolo              # YOLO26s + USB webcam
./run.sh rtdetr --csi      # RT-DETR-L + IMX219 CSI camera
./run.sh yolo --csi --sensor-id 1   # second CSI port
```

**Never run the detect scripts directly** without first sourcing the venv and exporting `LD_LIBRARY_PATH` — torch will fail to import otherwise (see Library Paths below).

If you need to run a script directly for debugging:
```bash
source venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
python3 detect_gpu.py --csi
```

---

## Library Paths — Critical Detail

`libcusparseLt.so.0` is absent from the Jetson system CUDA. It is provided by the pip package `nvidia-cusparselt-cu12`, located at:

```
venv/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0
```

`run.sh` exports **only this one directory** to `LD_LIBRARY_PATH`:

```bash
CUSPARSELT_LIB="$SCRIPT_DIR/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib"
export LD_LIBRARY_PATH="${CUSPARSELT_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

**Do not** add all pip `nvidia/*/lib` directories — that overrides system cuDNN 9.3 with pip cuDNN 9.1 and causes `CUBLAS_STATUS_ALLOC_FAILED`.

---

## Exact Working Dependency Versions

| Package | Version | Notes |
|---|---|---|
| `torch` | `2.5.0a0+872d972e41.nv24.08` | NVIDIA Jetson JP61 wheel — **not** from pytorch.org |
| `torchvision` | `0.20.1` | Installed with `--no-deps` to avoid torch overwrite |
| `numpy` | `1.26.4` | Must be `<2.0.0` — system cv2 requires numpy 1.x ABI |
| `ultralytics` | `8.4.41` | |
| `Pillow` | `12.2.0` | |
| `nvidia-cusparselt-cu12` | any | Provides `libcusparseLt.so.0` only |
| `opencv` | `4.8.0` (system) | From `/usr/lib/python3.10/dist-packages/cv2/` |

### Why the NVIDIA Jetson wheel for torch

Generic `torch+cu126` from pytorch.org does **not** include a native kernel for SM 8.7. It falls back to PTX JIT but the JIT path also fails with `GET was unable to find an engine`. The NVIDIA Jetson-native wheel at:

```
https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

is compiled for SM 8.7 and links against the system CUDA/cuDNN.

### Why system OpenCV, not pip

`opencv-python` from pip is built without GStreamer. The IMX219 CSI camera requires GStreamer via `nvarguscamerasrc`. The system OpenCV at `/usr/lib/python3.10/dist-packages/cv2/` has GStreamer 1.20.3 support.

The file `venv/lib/python3.10/site-packages/system_cv2.pth` adds the system dist-packages path to the venv. **Never install `opencv-python` into the venv** — it will shadow system cv2 and break CSI camera access.

---

## Active Patches in the Venv

Two torchvision files are patched due to ABI incompatibility between `torchvision 0.20.1` (compiled for `torch 2.5.1`) and the Jetson wheel (`torch 2.5.0a0`).

### Patch 1 — `venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py`

**Problem:** Jetson torch registers some torchvision ops as `CompositeImplicitAutograd`, so torchvision's import-time meta kernel registration raises `RuntimeError`.

**Fix:** `register_meta`'s inner wrapper catches and silences `RuntimeError`:
```python
try:
    get_meta_lib().impl(...)
except RuntimeError:
    pass
```
Meta kernels are only needed for `torch.compile` / FX tracing, not inference.

### Patch 2 — `venv/lib/python3.10/site-packages/torchvision/ops/boxes.py`

**Problem:** The CPU/CUDA dispatch key integers are swapped between torch 2.5.0a0 and 2.5.1, so `torch.ops.torchvision.nms` fails on both CPU and CUDA tensors.

**Fix:** A pure-PyTorch `_pure_torch_nms()` function is inserted above `nms()`, and `nms()` calls it instead of `torch.ops.torchvision.nms`. Works on any device without the C++ extension.

**If torchvision is ever reinstalled**, both patches must be reapplied. `setup.sh` does this automatically.

---

## CSI Camera Setup

The IMX219 is accessed via GStreamer's `nvarguscamerasrc` plugin (part of JetPack). The pipeline used:

```
nvarguscamerasrc sensor-id=0 !
video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 !
nvvidconv flip-method=0 !
video/x-raw, width=1280, height=720, format=BGRx !
videoconvert ! video/x-raw, format=BGR ! appsink drop=1
```

`cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` opens this. `cv2.CAP_GSTREAMER` is required — it will silently fail without it.

The `nvargus-daemon` system service must be running. `run.sh` restarts it before every launch to clear stale sessions from previous crashes. This is passwordless via `/etc/sudoers.d/nvargus-daemon`.

The horizontal mirror flip (`cv2.flip(frame, 1)`) is skipped for CSI — it is only applied for USB webcams.

---

## Debugging Common Errors

### `ImportError: libcusparseLt.so.0: cannot open shared object file`
`LD_LIBRARY_PATH` is not set. Use `run.sh` or export the path manually (see Library Paths above).

### `GET was unable to find an engine to execute this computation`
Wrong torch wheel is installed (pytorch.org generic instead of NVIDIA Jetson JP61). Reinstall:
```bash
source venv/bin/activate
pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### `RuntimeError: operator torchvision::nms does not exist` or `dets must be a CPU tensor` or `NotImplementedError: Could not run 'torchvision::nms'`
Patch 2 is missing from `torchvision/ops/boxes.py`. Run `setup.sh` or apply it manually.

### `RuntimeError: We should not register a meta kernel directly to the operator`
Patch 1 is missing from `torchvision/_meta_registrations.py`. Run `setup.sh` or apply it manually.

### `CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`
Too many pip nvidia lib paths are in `LD_LIBRARY_PATH`. This loads pip cuDNN 9.1 instead of system cuDNN 9.3. Only add the cusparseLt lib directory.

### `Error: Failed to create CaptureSession` (camera)
Stale argus session from a previous crash. Restart the daemon:
```bash
sudo systemctl restart nvargus-daemon
```
`run.sh` does this automatically on every launch.

### `GStreamer warning: Cannot query video position`
Harmless. GStreamer can't seek on a live camera stream.

### `NvMapMemAllocInternalTagged: error 12`
Unified memory pressure. Usually harmless for YOLO26s. For larger models (RT-DETR-L), reduce `imgsz` in the `model.predict()` call (e.g., 480 instead of 640) and add `half=True` for FP16 inference.

### Black screen / `select() timeout` on `/dev/video0`
The IMX219 CSI camera does not respond to V4L2. Always use `--csi` flag with `run.sh`.

### `torchvision X requires torch==Y, but you have torch Z`
pip dependency resolver warning — safe to ignore. The `--no-deps` install of torchvision prevents torch from being overwritten.

---

## File Structure

```
Real-Time-Collision-3/
├── detect_gpu.py          # YOLO26s + CUDA
├── detect_cpu.py          # YOLO26s + CPU
├── detect_rtdetr_gpu.py   # RT-DETR-L + CUDA
├── detect_rtdetr_cpu.py   # RT-DETR-L + CPU
├── run.sh                 # Entry point — selects variant, sets env, restarts daemon
├── setup.sh               # One-time environment setup + patches
├── requirements.txt       # pip packages (torch/torchvision installed separately)
├── SETUP.md               # Human-readable setup guide
├── CLAUDE.md              # This file
├── yolo26s.pt             # Downloaded on first run
└── rtdetr-l.pt            # Downloaded on first run
```

---

## What setup.sh Does (in order)

1. Installs `python3-tk python3-full` via apt
2. Adds passwordless sudoers rule for `nvargus-daemon` restart
3. Creates fresh venv using system Python (`/usr/bin/python3`)
4. Installs NVIDIA Jetson torch wheel
5. Installs torchvision 0.20.1 with `--no-deps`
6. Installs remaining packages from `requirements.txt`
7. Applies Patch 1 (`_meta_registrations.py`)
8. Applies Patch 2 (`ops/boxes.py`)
9. Creates `system_cv2.pth` to expose system OpenCV
10. Installs `nvidia-cusparselt-cu12`
