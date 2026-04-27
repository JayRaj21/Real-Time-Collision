#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure system tkinter is present (Homebrew Python lacks _tkinter)
echo "Installing system tkinter..."
sudo apt-get install -y python3-tk python3-full

# Allow any sudo-capable user to restart nvargus-daemon without a password,
# so the CSI camera session is always cleanly reset on launch.
SUDOERS_FILE="/etc/sudoers.d/nvargus-daemon"
SUDOERS_RULE="%sudo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart nvargus-daemon"
if ! sudo grep -qxF "$SUDOERS_RULE" "$SUDOERS_FILE" 2>/dev/null; then
    echo "$SUDOERS_RULE" | sudo tee "$SUDOERS_FILE" > /dev/null
    sudo chmod 0440 "$SUDOERS_FILE"
    echo "Passwordless nvargus-daemon restart configured."
fi

# Deactivate any active venv
deactivate 2>/dev/null || true

# Recreate venv using the system Python so _tkinter is available
rm -rf venv
/usr/bin/python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
# Install Jetson-native PyTorch (compiled for SM 8.7 / Orin, JetPack 6.x)
pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --no-deps
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126

# torchvision 0.20.1 tries to register meta kernels for ops that the Jetson
# torch 2.5.0a0 already registers as CompositeImplicitAutograd — patch it to
# silently skip those (meta kernels are only needed for torch.compile, not inference).
python3 - <<'PYEOF'
import re, pathlib
f = pathlib.Path("venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py")
src = f.read_text()
old = (
    "        if torchvision.extension._has_ops():\n"
    "            get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)\n"
)
new = (
    "        if torchvision.extension._has_ops():\n"
    "            try:\n"
    "                get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)\n"
    "            except RuntimeError:\n"
    "                pass\n"
)
if old in src:
    f.write_text(src.replace(old, new))
    print("Patched torchvision/_meta_registrations.py")
else:
    print("Patch already applied or not needed")
PYEOF

# torchvision 0.20.1 C++ dispatch is ABI-incompatible with torch 2.5.0a0
# (Jetson wheel): CPU/CUDA dispatch keys are swapped, so nms() fails on both
# devices. Replace with a pure-PyTorch NMS that needs no C++ extension.
python3 - <<'PYEOF'
import pathlib
f = pathlib.Path("venv/lib/python3.10/site-packages/torchvision/ops/boxes.py")
src = f.read_text()
if "_pure_torch_nms" in src:
    print("NMS patch already applied")
else:
    pure_nms = (
        "def _pure_torch_nms(boxes, scores, iou_threshold):\n"
        '    """Pure-PyTorch NMS — works on CPU and CUDA without the C++ extension."""\n'
        "    import torch\n"
        "    if boxes.numel() == 0:\n"
        "        return torch.empty((0,), dtype=torch.int64, device=boxes.device)\n"
        "    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]\n"
        "    areas = (x2 - x1) * (y2 - y1)\n"
        "    order = scores.argsort(descending=True)\n"
        "    keep = []\n"
        "    while order.numel() > 0:\n"
        "        i = order[0]\n"
        "        keep.append(i.item())\n"
        "        if order.numel() == 1:\n"
        "            break\n"
        "        rest = order[1:]\n"
        "        inter_w = (torch.minimum(x2[i], x2[rest]) - torch.maximum(x1[i], x1[rest])).clamp(min=0)\n"
        "        inter_h = (torch.minimum(y2[i], y2[rest]) - torch.maximum(y1[i], y1[rest])).clamp(min=0)\n"
        "        iou = (inter_w * inter_h) / (areas[i] + areas[rest] - inter_w * inter_h)\n"
        "        order = rest[iou <= iou_threshold]\n"
        "    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)\n"
        "\n\n"
    )
    old_call = "    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)\n"
    new_call = "    return _pure_torch_nms(boxes, scores, iou_threshold)\n"
    marker = "def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:\n"
    src = src.replace(marker, pure_nms + marker).replace(old_call, new_call)
    f.write_text(src)
    print("Patched torchvision/ops/boxes.py (pure-PyTorch NMS)")
PYEOF

# Use system OpenCV (has GStreamer/CSI support); pip's opencv-python does not.
# Add system dist-packages to the venv's path via a .pth file.
echo "/usr/lib/python3.10/dist-packages" > venv/lib/python3.10/site-packages/system_cv2.pth

# Install nvidia-cusparselt only (libcusparseLt.so.0 is missing from system CUDA).
# All other CUDA libs come from the system JetPack installation.
pip install nvidia-cusparselt-cu12 --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "Setup complete. Run the app with: ./run.sh"
echo "  USB webcam:  ./run.sh yolo"
echo "  CSI camera:  ./run.sh yolo --csi"
