#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Reset the CSI camera daemon so every launch starts with a clean session.
sudo systemctl restart nvargus-daemon

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODEL="${1:-yolo}"   # default: yolo   |   pass "rtdetr" for RT-DETR
shift || true        # consume $1 so remaining args (e.g. --csi) pass through

# libcusparseLt.so.0 is absent from Jetson's system CUDA; the pip nvidia package
# has it. Expose only that one directory — exposing all pip nvidia libs overrides
# system cuDNN/cuBLAS with wrong versions and breaks GPU execution.
CUSPARSELT_LIB="$SCRIPT_DIR/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib"
export LD_LIBRARY_PATH="${CUSPARSELT_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHON="$(command -v python3)"

has_cuda() {
    "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

case "$MODEL" in
    yolo)
        if has_cuda; then
            echo "CUDA detected — running YOLO26 GPU version."
            exec "$PYTHON" detect_gpu.py "$@"
        else
            echo "No CUDA GPU detected — running YOLO26 CPU version."
            exec "$PYTHON" detect_cpu.py "$@"
        fi
        ;;
    rtdetr)
        if has_cuda; then
            echo "CUDA detected — running RT-DETR GPU version."
            exec "$PYTHON" detect_rtdetr_gpu.py "$@"
        else
            echo "No CUDA GPU detected — running RT-DETR CPU version."
            echo "WARNING: RT-DETR on CPU will be very slow (~1-3 FPS)."
            exec "$PYTHON" detect_rtdetr_cpu.py "$@"
        fi
        ;;
    *)
        echo "Usage: $0 [yolo|rtdetr] [--csi] [--sensor-id N]"
        echo "  yolo   — YOLO26s  (default, fast)"
        echo "  rtdetr — RT-DETR-L (transformer, higher accuracy)"
        echo "  --csi  — use IMX219 CSI camera instead of USB webcam"
        exit 1
        ;;
esac
