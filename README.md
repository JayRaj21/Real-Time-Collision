# Jetson Collision Detector

Real-time object detection and collision identification on a **Jetson Orin Nano** with a **Waveshare IMX219-160 CSI camera**, using **Qwen2.5-VL-3B** as the vision-language model.

---

## Requirements

- Jetson Orin Nano 8 GB, JetPack 6.1 (L4T R36.4), Python 3.10
- CSI camera connected to CAM0
- X11 display (physical or forwarded)

---

## 1 — Install PyTorch (Jetson wheel)

Do **not** use `pip install torch` — the PyPI wheel has no CUDA support on aarch64.

```bash
# Install cusparselt (required by PyTorch 2.5+)
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
tar xf libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz
sudo cp libcusparse_lt-linux-aarch64-0.7.1.0-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig

# Install the Jetson PyTorch wheel for JetPack 6.1 / Python 3.10
pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

For other JetPack versions, find the correct wheel at:
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

---

## 2 — Install OpenCV (system package)

```bash
sudo apt install -y python3-opencv
```

Do **not** use `pip install opencv-python` — the PyPI wheel lacks GStreamer support needed for the CSI camera.

---

## 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

The Qwen2.5-VL-3B model weights (~6 GB) are downloaded automatically on first run.

---

## 4 — Run

```bash
python3 main.py
```

Over SSH with X forwarding:
```bash
ssh -X jetson@<jetson-ip>
export DISPLAY=:0
python3 main.py
```

---

## GUI

| Control | Function |
|---|---|
| **Show Boxes** | Toggle bounding boxes on/off |
| **FPS** | Display loop frame rate |
| **Contacts** | Active collision events |
| Ctrl-C / close window | Shutdown |

Collision events are always shown: a red line connects colliding objects with a `objA ↔ objB` label at the midpoint.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: Cannot open CSI camera` | Check ribbon cable in CAM0; run `gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink` |
| `torch.cuda.is_available()` returns `False` | PyPI torch installed — reinstall with the Jetson wheel (§1) |
| `ImportError: No module named 'cv2'` | Run `sudo apt install python3-opencv` |
| `_tkinter.TclError: no display` | Set `DISPLAY=:0` |
| Black canvas | Check `sudo systemctl status nvargus-daemon` and ribbon cable seating |
