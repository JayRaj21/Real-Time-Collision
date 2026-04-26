#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODEL="${1:-yolo}"   # default: yolo   |   pass "rtdetr" for RT-DETR

PYTHON="$(command -v python3)"

has_cuda() {
    "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

case "$MODEL" in
    yolo)
        if has_cuda; then
            echo "CUDA detected — running YOLO26 GPU version."
            exec "$PYTHON" detect_gpu.py
        else
            echo "No CUDA GPU detected — running YOLO26 CPU version."
            exec "$PYTHON" detect_cpu.py
        fi
        ;;
    rtdetr)
        if has_cuda; then
            echo "CUDA detected — running RT-DETR GPU version."
            exec "$PYTHON" detect_rtdetr_gpu.py
        else
            echo "No CUDA GPU detected — running RT-DETR CPU version."
            echo "WARNING: RT-DETR on CPU will be very slow (~1-3 FPS)."
            exec "$PYTHON" detect_rtdetr_cpu.py
        fi
        ;;
    *)
        echo "Usage: $0 [yolo|rtdetr]"
        echo "  yolo   — YOLO26s  (default, fast)"
        echo "  rtdetr — RT-DETR-L (transformer, higher accuracy)"
        exit 1
        ;;
esac
