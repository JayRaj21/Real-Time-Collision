import sys
import time
import queue
import threading
from collections import Counter, defaultdict, deque
import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_PATH = "yolo26s.pt"
CONF_THRESHOLD = 0.4
DEVICE = "cuda"

CANVAS_W, CANVAS_H = None, None
BG       = "#1a1a1a"
PANEL_BG = "#242424"
ACCENT   = "#00e676"
MUTED    = "#777777"
WHITE    = "#f0f0f0"

ROLLING_WINDOW = 30  # frames used for rolling quality estimates


# ---------------------------------------------------------------- helpers --

def _box_iou(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0


def _avg_pairwise_iou(xyxy: list) -> float:
    """Mean IoU across every pair of detected boxes in one frame."""
    if len(xyxy) < 2:
        return 0.0
    vals = [_box_iou(xyxy[i], xyxy[j])
            for i in range(len(xyxy)) for j in range(i + 1, len(xyxy))]
    return sum(vals) / len(vals)


def _ap_from_confs(confs: list) -> float:
    """
    Simplified AP: area under the precision curve when ranked by confidence.
    Each detection is treated as a true positive weighted by its score,
    approximating AP@0.5 without ground-truth labels.
    """
    if not confs:
        return 0.0
    ranked = sorted(confs, reverse=True)
    return sum((k + 1) * c for k, c in enumerate(ranked)) / (
        sum(range(1, len(ranked) + 1))
    )


def _boxes_overlap(b1, b2) -> bool:
    return b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]


def _find_collision_pairs(xyxy: list) -> list:
    return [(i, j) for i in range(len(xyxy))
            for j in range(i + 1, len(xyxy)) if _boxes_overlap(xyxy[i], xyxy[j])]


def _draw_collision_highlights(frame, xyxy: list, indices: set):
    out = frame.copy()
    for idx in indices:
        x1, y1, x2, y2 = map(int, xyxy[idx])
        cv2.rectangle(out, (x1, y1), (x2, y2), (30, 30, 255), 3)
        cv2.putText(out, "COLLISION", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------- rolling tracker -

class _RollingMetrics:
    """Accumulates per-frame stats and emits 30-frame rolling estimates."""

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        self._prec:     deque = deque(maxlen=window)
        self._has_det:  deque = deque(maxlen=window)
        self._class_ap: deque = deque(maxlen=window)

    def update(self, confs: list, class_ids: list) -> dict:
        has = len(confs) > 0
        frame_prec = sum(confs) / len(confs) if has else 0.0

        if has:
            by_class: dict = defaultdict(list)
            for c, cf in zip(class_ids, confs):
                by_class[c].append(cf)
            frame_map = sum(_ap_from_confs(v) for v in by_class.values()) / len(by_class)
        else:
            frame_map = 0.0

        self._prec.append(frame_prec)
        self._has_det.append(1.0 if has else 0.0)
        self._class_ap.append(frame_map)

        precision = sum(self._prec) / len(self._prec)
        recall    = sum(self._has_det) / len(self._has_det)
        denom     = precision + recall
        f1        = 2 * precision * recall / denom if denom > 0 else 0.0
        mAP       = sum(self._class_ap) / len(self._class_ap)

        return dict(precision=precision, recall=recall, f1=f1, mAP=mAP,
                    ap=_ap_from_confs(confs))


# --------------------------------------------------------------- app ------

class DetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO26 Object Detection — GPU")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.running = True
        self.collision_enabled = tk.BooleanVar(value=False)
        self._collision_flag = False
        self.collision_enabled.trace_add("write", lambda *_: self._sync_flag())

        self._frame_q:   queue.Queue = queue.Queue(maxsize=1)
        self._metrics_q: queue.Queue = queue.Queue(maxsize=1)
        self._rolling = _RollingMetrics()

        self.model = YOLO(MODEL_PATH)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam (device 0).")
            sys.exit(1)

        global CANVAS_W, CANVAS_H
        CANVAS_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        CANVAS_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            result = self.model.info(verbose=True)
            layers, params, _, gflops = result if result else (0, 0, 0, 0.0)
        except Exception:
            layers, params, gflops = 0, 0, 0.0
        self._info = dict(layers=layers, params=params, gflops=gflops)

        self._build_ui()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        self._refresh_ui()

    def _sync_flag(self):
        self._collision_flag = self.collision_enabled.get()

    # ---------------------------------------------------------------- UI --

    def _build_ui(self) -> None:
        # ── video ──────────────────────────────────────────────────────────
        left = tk.Frame(self.root, bg=BG)
        left.pack(side=tk.LEFT, padx=(12, 6), pady=12)
        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack()

        # ── scrollable side panel ──────────────────────────────────────────
        outer = tk.Frame(self.root, bg=PANEL_BG, width=240)
        outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 12), pady=12)
        outer.pack_propagate(False)

        canvas_scroll = tk.Canvas(outer, bg=PANEL_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient=tk.VERTICAL,
                                 command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(canvas_scroll, bg=PANEL_BG)
        canvas_scroll.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>",
                   lambda e: canvas_scroll.configure(
                       scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.bind("<MouseWheel>",
                           lambda e: canvas_scroll.yview_scroll(-1 * (e.delta // 120), "units"))
        canvas_scroll.bind("<Button-4>",
                           lambda e: canvas_scroll.yview_scroll(-1, "units"))
        canvas_scroll.bind("<Button-5>",
                           lambda e: canvas_scroll.yview_scroll(1, "units"))

        # -- Performance --
        self._section(right, "PERFORMANCE")
        self._mv_fps      = self._row(right, "FPS")
        self._mv_preproc  = self._row(right, "Preprocess")
        self._mv_infer    = self._row(right, "Inference")
        self._mv_postproc = self._row(right, "Postprocess")
        self._mv_total    = self._row(right, "Total latency")
        self._divider(right)

        # -- Detections --
        self._section(right, "DETECTIONS")
        self._mv_count    = self._row(right, "Count")
        self._mv_avg_conf = self._row(right, "Avg confidence")
        self._mv_max_conf = self._row(right, "Max confidence")
        self._divider(right)

        # -- Quality estimates --
        self._section(right, f"QUALITY  ({ROLLING_WINDOW}-frame rolling)")
        tk.Label(right, text="confidence-based estimates",
                 bg=PANEL_BG, fg="#555555",
                 font=("Helvetica", 7, "italic")).pack(anchor="w", padx=14)
        self._mv_precision = self._row(right, "Precision")
        self._mv_recall    = self._row(right, "Recall")
        self._mv_f1        = self._row(right, "F1 Score")
        self._mv_iou       = self._row(right, "Avg IoU")
        self._mv_ap        = self._row(right, "AP (frame)")
        self._mv_map       = self._row(right, "mAP (rolling)")
        self._divider(right)

        # -- Collision --
        self._section(right, "COLLISION DETECTION")
        tog = tk.Frame(right, bg=PANEL_BG)
        tog.pack(fill=tk.X, padx=14, pady=(2, 4))
        tk.Checkbutton(tog, text="Enable", variable=self.collision_enabled,
                       bg=PANEL_BG, fg=WHITE, selectcolor="#444",
                       activebackground=PANEL_BG, activeforeground=WHITE,
                       font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._mv_col_pairs = self._row(right, "Pairs detected")
        self._divider(right)

        # -- Classes --
        self._section(right, "CLASSES DETECTED")
        self._classes_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self._classes_var, bg=PANEL_BG, fg=ACCENT,
                 font=("Courier", 10), justify=tk.LEFT,
                 wraplength=200).pack(anchor="w", padx=14, pady=(0, 4))
        self._divider(right)

        # -- Model info --
        self._section(right, "MODEL INFO")
        self._row(right, "Name",    MODEL_PATH,  static=True)
        self._row(right, "Device",  torch.cuda.get_device_name(0), static=True)
        self._row(right, "Layers",  str(self._info["layers"]),  static=True)
        self._row(right, "Params",  f"{self._info['params']/1e6:.1f} M", static=True)
        self._row(right, "GFLOPs", f"{self._info['gflops']:.1f}", static=True)

        tk.Button(right, text="Quit", command=self._on_close,
                  bg="#c62828", fg=WHITE, font=("Helvetica", 11, "bold"),
                  activebackground="#b71c1c", activeforeground=WHITE,
                  relief=tk.FLAT, cursor="hand2",
                  padx=12, pady=6).pack(pady=14)

    def _section(self, parent, title: str) -> None:
        tk.Label(parent, text=title, bg=PANEL_BG, fg=MUTED,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=14, pady=(10, 2))

    def _divider(self, parent) -> None:
        tk.Frame(parent, bg="#3a3a3a", height=1).pack(fill=tk.X, padx=10, pady=4)

    def _row(self, parent, label: str, initial: str = "—",
             static: bool = False) -> tk.StringVar:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(row, text=label, bg=PANEL_BG, fg=MUTED,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        var = tk.StringVar(value=initial)
        tk.Label(row, textvariable=var, bg=PANEL_BG,
                 fg=MUTED if static else ACCENT,
                 font=("Helvetica", 10, "bold")).pack(side=tk.RIGHT)
        return var

    # --------------------------------------------------------- inference --

    def _inference_loop(self) -> None:
        prev = time.perf_counter()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            results = self.model.predict(
                frame, device=DEVICE, conf=CONF_THRESHOLD, imgsz=640, verbose=False
            )
            r = results[0]
            annotated = r.plot()

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            boxes = r.boxes
            n = len(boxes)
            confs      = boxes.conf.tolist() if n > 0 else []
            class_ids  = boxes.cls.int().tolist() if n > 0 else []
            xyxy       = boxes.xyxy.tolist() if n > 0 else []
            avg_conf   = sum(confs) / len(confs) if confs else 0.0
            max_conf   = max(confs) if confs else 0.0

            class_names = r.names
            counts = Counter(class_names[c] for c in class_ids)
            classes_str = "\n".join(f"{v}× {k}" for k, v in sorted(counts.items())) or "—"

            quality = self._rolling.update(confs, class_ids)
            quality["iou"] = _avg_pairwise_iou(xyxy)

            collision_pairs = 0
            if self._collision_flag and n > 1:
                pairs = _find_collision_pairs(xyxy)
                collision_pairs = len(pairs)
                if pairs:
                    colliding = {idx for p in pairs for idx in p}
                    annotated = _draw_collision_highlights(annotated, xyxy, colliding)

            speed = r.speed
            self._put(self._frame_q, annotated)
            self._put(self._metrics_q, dict(
                fps=fps,
                preproc=speed.get("preprocess", 0),
                infer=speed.get("inference", 0),
                postproc=speed.get("postprocess", 0),
                n=n, avg_conf=avg_conf, max_conf=max_conf,
                classes_str=classes_str,
                collision_pairs=collision_pairs,
                **quality,
            ))

    @staticmethod
    def _put(q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    # ------------------------------------------------------- UI refresh --

    def _refresh_ui(self) -> None:
        if not self.running:
            return

        try:
            frame = self._frame_q.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((CANVAS_W, CANVAS_H), Image.BILINEAR)
            photo = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        except queue.Empty:
            pass

        try:
            m = self._metrics_q.get_nowait()
            total = m["preproc"] + m["infer"] + m["postproc"]
            has = m["n"] > 0

            self._mv_fps.set(f"{m['fps']:.1f}")
            self._mv_preproc.set(f"{m['preproc']:.1f} ms")
            self._mv_infer.set(f"{m['infer']:.1f} ms")
            self._mv_postproc.set(f"{m['postproc']:.1f} ms")
            self._mv_total.set(f"{total:.1f} ms")

            self._mv_count.set(str(m["n"]))
            self._mv_avg_conf.set(f"{m['avg_conf']:.3f}" if has else "—")
            self._mv_max_conf.set(f"{m['max_conf']:.3f}" if has else "—")

            self._mv_precision.set(f"{m['precision']:.3f}")
            self._mv_recall.set(f"{m['recall']:.3f}")
            self._mv_f1.set(f"{m['f1']:.3f}")
            self._mv_iou.set(f"{m['iou']:.3f}" if m["n"] > 1 else "—")
            self._mv_ap.set(f"{m['ap']:.3f}" if has else "—")
            self._mv_map.set(f"{m['mAP']:.3f}")

            self._classes_var.set(m["classes_str"])
            self._mv_col_pairs.set(
                str(m["collision_pairs"]) if self.collision_enabled.get() else "—"
            )
        except queue.Empty:
            pass

        self.root.after(16, self._refresh_ui)

    # ---------------------------------------------------------------- quit -

    def _on_close(self) -> None:
        self.running = False
        self.cap.release()
        self.root.destroy()


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system.")
        print("Please run detect_cpu.py instead.")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    root = tk.Tk()
    DetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
