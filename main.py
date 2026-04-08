#!/usr/bin/env python3
"""
Jetson Orin Nano — real-time object detection & collision/contact detector
Camera : Waveshare IMX219-160 CSI via GStreamer (nvarguscamerasrc)
Model  : YOLOv8n (ultralytics) on CUDA
GUI    : Tkinter / X11
"""

import os
import sys
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

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "[ERROR] ultralytics is not installed.\n"
        "        Run:  pip install ultralytics"
    )

import torch

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

YOLO_SIZE   = 640       # width & height fed to YOLO (square crop)
DISP_W      = 960       # Tkinter canvas width
DISP_H      = 540       # Tkinter canvas height
CONF_THRESH = 0.40      # minimum detection confidence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Type alias
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
    """
    Return True when two axis-aligned bounding boxes overlap or share an edge.
    Boxes are (x_min, y_min, x_max, y_max).  Touching edges count as contact.
    """
    # Strict < means equal edges (touching) fall through → True
    return not (
        b1[2] < b2[0] or b2[2] < b1[0] or
        b1[3] < b2[1] or b2[3] < b1[1]
    )


def find_collisions(dets: List[Detection]) -> List[Tuple[int, int]]:
    """Return all (i, j) index pairs whose bounding boxes touch or overlap."""
    return [
        (i, j)
        for i in range(len(dets))
        for j in range(i + 1, len(dets))
        if boxes_touch(dets[i]["box"], dets[j]["box"])
    ]


# ── Camera thread ─────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """
    Reads frames from the CSI camera as fast as possible and caches the
    most recent one.  The display and inference threads fetch it via latest().
    """

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
    Pulls frames from in_queue, runs YOLOv8n on CUDA, and pushes
    (detections, collisions) into out_queue.  Keeps only the freshest result.
    """

    def __init__(self, model: YOLO,
                 in_queue: "queue.Queue[np.ndarray]",
                 out_queue: "queue.Queue[Tuple[List[Detection], List[Tuple[int,int]]]]"
                 ) -> None:
        super().__init__(daemon=True)
        self._model     = model
        self._in_queue  = in_queue
        self._out_queue = out_queue
        self._stop_evt  = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                frame = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            orig_h, orig_w = frame.shape[:2]

            # Resize to YOLO_SIZE × YOLO_SIZE (spec requirement)
            small   = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            results = self._model(small, device=DEVICE,
                                  conf=CONF_THRESH, verbose=False)[0]

            # Scale factors from YOLO space back to original frame space
            sx = orig_w / YOLO_SIZE
            sy = orig_h / YOLO_SIZE

            dets: List[Detection] = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "label": results.names[int(box.cls[0])],
                    "box":   (int(x1 * sx), int(y1 * sy),
                              int(x2 * sx), int(y2 * sy)),
                    "conf":  float(box.conf[0]),
                })

            payload = (dets, find_collisions(dets))

            # Discard stale result before publishing the fresh one
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
    """
    Annotate a frame with:
      • Bounding boxes (green = normal, red = in collision) when show_boxes=True
      • A red line between colliding objects' centres
      • A contact label "objA ↔ objB" at the midpoint of the line
    """
    out = frame.copy()
    hot_indices = {idx for pair in collisions for idx in pair}

    # ── Per-object boxes ──────────────────────────────────────────────────────
    if show_boxes:
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = d["box"]
            color = (0, 0, 230) if idx in hot_indices else (30, 200, 30)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label_txt = f"{d['label']}  {d['conf']:.2f}"
            cv2.putText(out, label_txt, (x1, max(y1 - 7, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # ── Collision overlays (drawn regardless of show_boxes) ───────────────────
    for i, j in collisions:
        d1, d2 = dets[i], dets[j]

        # Centres of each box
        cx1 = (d1["box"][0] + d1["box"][2]) // 2
        cy1 = (d1["box"][1] + d1["box"][3]) // 2
        cx2 = (d2["box"][0] + d2["box"][2]) // 2
        cy2 = (d2["box"][1] + d2["box"][3]) // 2

        # Red connecting line
        cv2.line(out, (cx1, cy1), (cx2, cy2), (0, 0, 220), 2, cv2.LINE_AA)

        # Contact label at midpoint
        mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
        label = f"{d1['label']} \u2194 {d2['label']}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 5
        # Dark background pill for readability
        cv2.rectangle(
            out,
            (mx - pad,      my - th - pad),
            (mx + tw + pad, my + baseline + pad),
            (15, 15, 15),
            cv2.FILLED,
        )
        cv2.putText(out, label, (mx, my),
                    font, font_scale, (50, 80, 255), thickness, cv2.LINE_AA)

    return out


# ── Tkinter application ───────────────────────────────────────────────────────

class App:
    """Main Tkinter window embedding the live annotated camera feed."""

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
        self._photo: Optional[ImageTk.PhotoImage] = None  # prevent GC
        self._dets: List[Detection] = []
        self._cols: List[Tuple[int, int]] = []
        self._fps_buf: List[float] = []

        self.show_boxes = tk.BooleanVar(value=True)

        # ── Window setup ──────────────────────────────────────────────────────
        root.title("Jetson Collision Detector")
        root.configure(bg="#111111")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Top status bar ────────────────────────────────────────────────────
        bar = tk.Frame(root, bg="#1a1a2e", pady=6)
        bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            bar, text="Jetson Collision Detector",
            bg="#1a1a2e", fg="#e0e0e0",
            font=("Helvetica", 13, "bold"),
        ).pack(side=tk.LEFT, padx=12)

        # Right-aligned controls (pack right → left)
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

        # Toggle switch for bounding boxes
        toggle = tk.Checkbutton(
            bar,
            text="Show Boxes",
            variable=self.show_boxes,
            bg="#1a1a2e", fg="#cccccc",
            selectcolor="#2e2e5e",
            activebackground="#1a1a2e",
            activeforeground="#ffffff",
            font=("Helvetica", 11),
            cursor="hand2",
        )
        toggle.pack(side=tk.RIGHT, padx=10)

        device_badge_color = "#00c853" if DEVICE == "cuda" else "#ff6f00"
        device_badge_text  = f"  {DEVICE.upper()}  "
        tk.Label(
            bar, text=device_badge_text,
            bg=device_badge_color, fg="#000000",
            font=("Courier", 10, "bold"),
            relief=tk.FLAT, padx=2,
        ).pack(side=tk.LEFT, padx=6)

        # ── Camera canvas ─────────────────────────────────────────────────────
        self._canvas = tk.Canvas(
            root,
            width=DISP_W, height=DISP_H,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack()

        # ── Bottom status strip ───────────────────────────────────────────────
        self._status_bar = tk.Label(
            root,
            text="Initialising camera...",
            bg="#0d0d0d", fg="#555555",
            font=("Courier", 9),
            anchor=tk.W, padx=8,
        )
        self._status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._update()

    # ── Display loop (runs in Tkinter / main thread) ──────────────────────────

    def _update(self) -> None:
        if not self._running:
            return

        frame = self._cam.latest()

        if frame is not None:
            # Feed inference thread — non-blocking, drop if busy
            if self._infer_in.empty():
                try:
                    self._infer_in.put_nowait(frame)
                except queue.Full:
                    pass

            # Consume latest inference result (non-blocking)
            try:
                self._dets, self._cols = self._infer_out.get_nowait()
            except queue.Empty:
                pass

            # Annotate frame and push to canvas
            vis = draw_results(frame, self._dets, self._cols,
                               self.show_boxes.get())
            vis = cv2.resize(vis, (DISP_W, DISP_H))
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

            n_obj = len(self._dets)
            n_col = len(self._cols)
            self._status_bar.config(
                text=f"Objects detected: {n_obj}   |   "
                     f"Active contacts: {n_col}   |   "
                     f"YOLO input: {YOLO_SIZE}×{YOLO_SIZE}   |   "
                     f"Display: {DISP_W}×{DISP_H}"
            )

        # FPS counter (rolling 1-second window)
        now = time.monotonic()
        self._fps_buf.append(now)
        self._fps_buf = [t for t in self._fps_buf if now - t < 1.0]
        self._fps_lbl.config(text=f"FPS: {len(self._fps_buf):3d}")
        self._col_lbl.config(text=f"Contacts: {len(self._cols)}")

        # Schedule next tick — 1 ms delay lets Tk drain events while still
        # running as fast as the camera / inference allows (≥20 FPS target)
        self._root.after(1, self._update)

    def _on_close(self) -> None:
        self._running = False
        self._root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure X11 is targeted (Tkinter on Linux uses X11 by default;
    # this is an explicit guard against accidental Wayland sessions)
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    # ── Open CSI camera — hard failure, no silent USB fallback ────────────────
    print(f"[init] Opening CSI camera ...")
    cap = open_camera()  # raises RuntimeError with diagnostics on failure
    print(f"[init] Camera opened OK  ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    # ── Load YOLOv8s (TensorRT engine if available, otherwise .pt) ───────────
    # yolov8s balances accuracy and efficiency well on Jetson Orin Nano.
    # Export to TensorRT for ~2× speedup:
    #   yolo export model=yolov8s.pt format=engine device=0
    engine_path = "yolov8s.engine"
    pt_path     = "yolov8s.pt"
    model_path  = engine_path if os.path.exists(engine_path) else pt_path
    print(f"[init] Loading {model_path} on device={DEVICE} ...")
    model = YOLO(model_path)
    if DEVICE == "cuda" and not model_path.endswith(".engine"):
        model.to("cuda")
    # Warm-up pass to compile CUDA kernels before the GUI opens
    dummy = np.zeros((YOLO_SIZE, YOLO_SIZE, 3), dtype=np.uint8)
    model(dummy, device=DEVICE, conf=CONF_THRESH, verbose=False)
    print("[init] Model ready.")

    # ── Start background threads ──────────────────────────────────────────────
    cam_thread = CameraThread(cap)
    cam_thread.start()

    infer_in:  "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
    infer_out: "queue.Queue[Tuple]"      = queue.Queue(maxsize=2)
    infer_thread = InferenceThread(model, infer_in, infer_out)
    infer_thread.start()

    # ── Launch GUI ────────────────────────────────────────────────────────────
    root = tk.Tk()
    # Verify windowing system is X11 (not Wayland / other)
    winsys = root.tk.call("tk", "windowingsystem")
    if winsys != "x11":
        print(f"[WARNING] Unexpected windowing system: '{winsys}'. "
              "This app targets X11.  Set DISPLAY correctly if running over SSH.",
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
