#!/usr/bin/env python3
"""
Jetson Orin Nano — real-time object collision detector  (v2 — VLM edition)
Camera : Waveshare IMX219-160 CSI via GStreamer (nvarguscamerasrc)
Model  : SmolVLM-Instruct (HuggingFaceTB/SmolVLM-Instruct, ~2.2B params)
GUI    : Tkinter / X11

Key difference from v1:
  v1 used YOLOv8 (pure object-detection CNN, ~30+ FPS on Jetson GPU).
  v2 uses SmolVLM-Instruct, a 2.2B-parameter Vision-Language Model that
  fits in ~5 GB — within the Jetson Orin Nano's 8 GB unified memory.
  It is prompted to return detected objects as structured JSON bounding
  boxes, so the downstream collision logic is identical to v1.
  Expected throughput: 1–3 FPS on Jetson Orin Nano (GPU, float16).
"""

import os
import sys
import json
import re
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple

# ── Force X11 display before any Tkinter import ───────────────────────────────
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch

# ── Torchvision stub ──────────────────────────────────────────────────────────
# Qwen2VLImageProcessor imports torchvision.transforms.v2.functional only for
# the InterpolationMode type hint.  No pre-built torchvision wheel exists for
# JetPack 6.1 aarch64, so we provide a minimal stub that satisfies the import.
import types as _types, enum as _enum, importlib.machinery as _imm
if "torchvision" not in sys.modules:
    def _make_mod(name):
        m = _types.ModuleType(name)
        m.__spec__    = _imm.ModuleSpec(name, loader=_imm.SourceFileLoader(name, ""), origin="stub")
        m.__file__    = "<torchvision-stub>"
        m.__version__ = "0.20.0"
        m.__loader__  = None
        m.__package__ = name.rsplit(".", 1)[0]
        return m

    class _InterpolationMode(_enum.Enum):
        NEAREST = 0; NEAREST_EXACT = 8
        BILINEAR = 2; BICUBIC = 3
        BOX = 4; HAMMING = 5; LANCZOS = 1

    # Build all submodules transformers may import from torchvision
    _tv             = _make_mod("torchvision")
    _tv_io          = _make_mod("torchvision.io")
    _tv_ops         = _make_mod("torchvision.ops")
    _tv_ops_boxes   = _make_mod("torchvision.ops.boxes")
    _tv_misc        = _make_mod("torchvision.misc")
    _tv_misc_ops    = _make_mod("torchvision.misc.ops")
    _tv2            = _make_mod("torchvision.transforms")
    _tv2f           = _make_mod("torchvision.transforms.functional")
    _tv2v           = _make_mod("torchvision.transforms.v2")
    _tv2vf          = _make_mod("torchvision.transforms.v2.functional")

    # Populate InterpolationMode wherever it's accessed
    for _m in (_tv2, _tv2f, _tv2v, _tv2vf):
        _m.InterpolationMode = _InterpolationMode

    # Wire up the module tree
    _tv.io          = _tv_io
    _tv.ops         = _tv_ops
    _tv_ops.boxes   = _tv_ops_boxes
    _tv.misc        = _tv_misc
    _tv_misc.ops    = _tv_misc_ops
    _tv.transforms  = _tv2
    _tv2.functional = _tv2f
    _tv2.v2         = _tv2v
    _tv2v.functional = _tv2vf

    sys.modules.update({
        "torchvision":                          _tv,
        "torchvision.io":                       _tv_io,
        "torchvision.ops":                      _tv_ops,
        "torchvision.ops.boxes":                _tv_ops_boxes,
        "torchvision.misc":                     _tv_misc,
        "torchvision.misc.ops":                 _tv_misc_ops,
        "torchvision.transforms":               _tv2,
        "torchvision.transforms.functional":    _tv2f,
        "torchvision.transforms.v2":            _tv2v,
        "torchvision.transforms.v2.functional": _tv2vf,
    })

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
except ImportError as e:
    sys.exit(
        f"[ERROR] transformers import failed: {e}\n"
        "        Run:  pip install 'transformers>=5.0.0'"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

GST_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=1"
)

# Resize frames before sending to the VLM — smaller = faster inference
VLM_W       = 640
VLM_H       = 640
DISP_W      = 960
DISP_H      = 540

MODEL_ID    = "HuggingFaceTB/SmolVLM-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float16 if DEVICE == "cuda" else torch.float32

# Prompt instructs the model to return JSON bounding boxes
DETECT_PROMPT = (
    "Detect all objects in this image. "
    "For each object return a JSON object with keys: "
    "\"label\" (string) and \"bbox_2d\" ([x1, y1, x2, y2] in pixel coordinates). "
    "Return only a JSON array, no extra text."
)

# Type alias (conf stored as 1.0 — SmolVLM does not output confidence scores)
Detection = Dict  # keys: label (str), box (x1,y1,x2,y2 ints), conf (float)


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    """Open the CSI camera via GStreamer. Raises RuntimeError on failure."""
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "\n[ERROR] Cannot open the CSI camera via GStreamer.\n"
            "\n  Possible causes:\n"
            "    • GStreamer Jetson plugins (nvarguscamerasrc) are not installed\n"
            "      or the package 'gstreamer1.0-plugins-bad' is missing.\n"
            "    • The CSI ribbon cable is not connected or not seated correctly.\n"
            "    • The camera is not enabled in /boot/extlinux/extlinux.conf or\n"
            "      the Jetson device-tree overlay.\n"
            "    • JetPack / L4T is not installed correctly.\n"
            "\n  Diagnostic commands:\n"
            "    gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink\n"
            "    v4l2-ctl --list-devices\n"
            "    ls /dev/video*\n"
            "\n  Do NOT fall back to a USB camera — fix the CSI pipeline first.\n"
        )
    return cap


# ── Collision detection ───────────────────────────────────────────────────────

def boxes_touch(b1: Tuple[int, int, int, int],
                b2: Tuple[int, int, int, int]) -> bool:
    return not (
        b1[2] < b2[0] or b2[2] < b1[0] or
        b1[3] < b2[1] or b2[3] < b1[1]
    )


def find_collisions(dets: List[Detection]) -> List[Tuple[int, int]]:
    return [
        (i, j)
        for i in range(len(dets))
        for j in range(i + 1, len(dets))
        if boxes_touch(dets[i]["box"], dets[j]["box"])
    ]


# ── SmolVLM inference helpers ─────────────────────────────────────────────────

def load_model() -> Tuple["AutoModelForVision2Seq", "AutoProcessor"]:
    """
    Download (first run, ~4 GB) and load SmolVLM-Instruct onto DEVICE.
    Subsequent runs load from ~/.cache/huggingface/hub/.
    """
    print(f"[init] Loading {MODEL_ID} on device={DEVICE}, dtype={DTYPE} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    return model, processor


def parse_detections(raw: str, orig_w: int, orig_h: int) -> List[Detection]:
    """
    Extract a JSON array of {label, bbox_2d} objects from the model's raw text.
    Bounding box coordinates are assumed to be in VLM_W × VLM_H pixel space and
    are scaled back to the original frame dimensions.

    Falls back gracefully to an empty list if parsing fails.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Find the first [...] array in the output
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    sx = orig_w / VLM_W
    sy = orig_h / VLM_H

    dets: List[Detection] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label", "object")
        bbox  = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]
        dets.append({
            "label": str(label),
            "box":   (int(x1 * sx), int(y1 * sy),
                      int(x2 * sx), int(y2 * sy)),
            "conf":  1.0,
        })
    return dets


def run_vlm_od(
    model:     "AutoModelForVision2Seq",
    processor: "AutoProcessor",
    frame_bgr: np.ndarray,
) -> List[Detection]:
    """
    Run SmolVLM object detection on a BGR numpy frame.
    Returns a list of Detection dicts with bounding boxes in original frame pixels.
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    small_rgb = cv2.cvtColor(
        cv2.resize(frame_bgr, (VLM_W, VLM_H)), cv2.COLOR_BGR2RGB
    )
    pil_img = Image.fromarray(small_rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DETECT_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=[pil_img],
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len  = inputs["input_ids"].shape[1]
    raw_output = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return parse_detections(raw_output, orig_w, orig_h)


# ── Camera thread ─────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame


# ── Inference thread ──────────────────────────────────────────────────────────

class InferenceThread(threading.Thread):
    """
    Pulls frames from in_queue, runs SmolVLM on GPU, and pushes
    (detections, collisions) into out_queue.

    Qwen2.5-VL is a generative VLM — inference is slower than YOLO
    (~1–4 FPS on Jetson Orin Nano).  The GUI displays the last known
    detections while the model processes the next frame.
    """

    def __init__(
        self,
        model:     "Qwen2_5_VLForConditionalGeneration",
        processor: "AutoProcessor",
        in_queue:  "queue.Queue[np.ndarray]",
        out_queue: "queue.Queue[Tuple[List[Detection], List[Tuple[int,int]]]]",
    ) -> None:
        super().__init__(daemon=True)
        self._model     = model
        self._processor = processor
        self._in_queue  = in_queue
        self._out_queue = out_queue
        self._stop_evt  = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                frame = self._in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            dets    = run_vlm_od(self._model, self._processor, frame)
            payload = (dets, find_collisions(dets))

            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except queue.Empty:
                    break
            self._out_queue.put(payload)


# ── Frame annotation ──────────────────────────────────────────────────────────

def draw_results(frame: np.ndarray,
                 dets: List[Detection],
                 collisions: List[Tuple[int, int]],
                 show_boxes: bool) -> np.ndarray:
    out = frame.copy()
    hot_indices = {idx for pair in collisions for idx in pair}

    if show_boxes:
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = d["box"]
            color = (0, 0, 230) if idx in hot_indices else (30, 200, 30)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(out, d["label"], (x1, max(y1 - 7, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for i, j in collisions:
        d1, d2 = dets[i], dets[j]
        cx1 = (d1["box"][0] + d1["box"][2]) // 2
        cy1 = (d1["box"][1] + d1["box"][3]) // 2
        cx2 = (d2["box"][0] + d2["box"][2]) // 2
        cy2 = (d2["box"][1] + d2["box"][3]) // 2

        cv2.line(out, (cx1, cy1), (cx2, cy2), (0, 0, 220), 2, cv2.LINE_AA)

        mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
        label = f"{d1['label']} \u2194 {d2['label']}"
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (tw, lh), baseline = cv2.getTextSize(label, font, fs, th)
        pad = 5
        cv2.rectangle(out,
                      (mx - pad,      my - lh - pad),
                      (mx + tw + pad, my + baseline + pad),
                      (15, 15, 15), cv2.FILLED)
        cv2.putText(out, label, (mx, my), font, fs, (50, 80, 255), th, cv2.LINE_AA)

    return out


# ── Tkinter application ───────────────────────────────────────────────────────

class App:
    def __init__(
        self,
        root: tk.Tk,
        cam_thread: CameraThread,
        infer_in:  "queue.Queue[np.ndarray]",
        infer_out: "queue.Queue[Tuple[List[Detection], List[Tuple[int,int]]]]",
    ) -> None:
        self._root      = root
        self._cam       = cam_thread
        self._infer_in  = infer_in
        self._infer_out = infer_out
        self._running   = True
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._dets: List[Detection] = []
        self._cols: List[Tuple[int, int]] = []
        self._fps_buf: List[float] = []
        self._infer_busy = False

        self.show_boxes = tk.BooleanVar(value=True)

        root.title("Jetson Collision Detector  [v2 — SmolVLM]")
        root.configure(bg="#111111")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        bar = tk.Frame(root, bg="#1a1a2e", pady=6)
        bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            bar, text="Jetson Collision Detector  [SmolVLM]",
            bg="#1a1a2e", fg="#e0e0e0",
            font=("Helvetica", 13, "bold"),
        ).pack(side=tk.LEFT, padx=12)

        self._fps_lbl = tk.Label(
            bar, text="FPS: --",
            bg="#1a1a2e", fg="#00e5ff",
            font=("Courier", 11, "bold"),
        )
        self._fps_lbl.pack(side=tk.RIGHT, padx=12)

        self._col_lbl = tk.Label(
            bar, text="Contacts: 0",
            bg="#1a1a2e", fg="#ff6d00",
            font=("Courier", 11, "bold"),
        )
        self._col_lbl.pack(side=tk.RIGHT, padx=12)

        tk.Checkbutton(
            bar, text="Show Boxes", variable=self.show_boxes,
            bg="#1a1a2e", fg="#cccccc", selectcolor="#2e2e5e",
            activebackground="#1a1a2e", activeforeground="#ffffff",
            font=("Helvetica", 11), cursor="hand2",
        ).pack(side=tk.RIGHT, padx=10)

        device_badge_color = "#00c853" if DEVICE == "cuda" else "#ff6f00"
        tk.Label(
            bar, text=f"  {DEVICE.upper()}  ",
            bg=device_badge_color, fg="#000000",
            font=("Courier", 10, "bold"), relief=tk.FLAT, padx=2,
        ).pack(side=tk.LEFT, padx=6)

        tk.Label(
            bar, text="  SmolVLM  ",
            bg="#006064", fg="#ffffff",
            font=("Courier", 10, "bold"), relief=tk.FLAT, padx=2,
        ).pack(side=tk.LEFT, padx=4)

        self._canvas = tk.Canvas(
            root, width=DISP_W, height=DISP_H,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack()

        self._status_bar = tk.Label(
            root, text="Initialising camera...",
            bg="#0d0d0d", fg="#555555",
            font=("Courier", 9), anchor=tk.W, padx=8,
        )
        self._status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._update()

    def _update(self) -> None:
        if not self._running:
            return

        frame = self._cam.latest()

        if frame is not None:
            if self._infer_in.empty():
                try:
                    self._infer_in.put_nowait(frame)
                except queue.Full:
                    pass

            try:
                self._dets, self._cols = self._infer_out.get_nowait()
                self._infer_busy = False
            except queue.Empty:
                self._infer_busy = True

            vis = draw_results(frame, self._dets, self._cols,
                               self.show_boxes.get())

            if self._infer_busy:
                cv2.putText(
                    vis, "VLM processing...",
                    (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
                )

            vis = cv2.resize(vis, (DISP_W, DISP_H))
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

            self._status_bar.config(
                text=f"Objects: {len(self._dets)}   |   "
                     f"Contacts: {len(self._cols)}   |   "
                     f"VLM input: {VLM_W}×{VLM_H}   |   "
                     f"Display: {DISP_W}×{DISP_H}   |   "
                     f"Model: {MODEL_ID}"
            )

        now = time.monotonic()
        self._fps_buf.append(now)
        self._fps_buf = [t for t in self._fps_buf if now - t < 1.0]
        self._fps_lbl.config(text=f"FPS: {len(self._fps_buf):3d}")
        self._col_lbl.config(text=f"Contacts: {len(self._cols)}")

        self._root.after(1, self._update)

    def _on_close(self) -> None:
        self._running = False
        self._root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    print("[init] Opening CSI camera ...")
    cap = open_camera()
    print(f"[init] Camera opened OK  ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    model, processor = load_model()

    print("[init] Running warm-up inference ...")
    dummy_bgr = np.zeros((VLM_H, VLM_W, 3), dtype=np.uint8)
    run_vlm_od(model, processor, dummy_bgr)
    print("[init] Model ready.")

    cam_thread = CameraThread(cap)
    cam_thread.start()

    infer_in:  "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
    infer_out: "queue.Queue[Tuple]"      = queue.Queue(maxsize=1)
    infer_thread = InferenceThread(model, processor, infer_in, infer_out)
    infer_thread.start()

    root = tk.Tk()
    winsys = root.tk.call("tk", "windowingsystem")
    if winsys != "x11":
        print(f"[WARNING] Unexpected windowing system: '{winsys}'.",
              file=sys.stderr)

    app = App(root, cam_thread, infer_in, infer_out)  # noqa: F841

    print("[init] GUI started.  Close the window or press Ctrl-C to quit.")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        print("[shutdown] Stopping threads ...")
        cam_thread.stop()
        infer_thread.stop()
        cam_thread.join(timeout=2.0)
        infer_thread.join(timeout=2.0)
        cap.release()
        print("[shutdown] Done.")


if __name__ == "__main__":
    main()
