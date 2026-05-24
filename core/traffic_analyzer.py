"""
Traffic Analyzer — Ultralytics YOLO11 with full GPU + TensorRT support.

Optimised for: Ryzen 5 5600X + RTX 3050 (8 GB VRAM)
Recommended model: YOLO11l with TensorRT FP16 → ~90–150 FPS
Default fallback:  YOLO11m PyTorch FP16 → ~80–130 FPS

Model auto-downloads from Ultralytics CDN on first use.
TensorRT engine is compiled once and cached as models/<name>_fp16.engine.

Falls back to YOLOv3 cv2.dnn if ultralytics is not installed.
"""

import os
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional, Callable

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── COCO class IDs for vehicles (same across all Ultralytics models) ──────────
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Available models ──────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    # YOLO11 — recommended (latest Ultralytics, 2024–2025)
    "yolo11n": ("yolo11n.pt", "YOLO11 Nano   ~5 MB   CPU-friendly, ~180 FPS GPU"),
    "yolo11s": ("yolo11s.pt", "YOLO11 Small  ~19 MB  fast, ~120 FPS GPU"),
    "yolo11m": ("yolo11m.pt", "YOLO11 Medium ~40 MB  balanced, ~80 FPS GPU"),
    "yolo11l": ("yolo11l.pt", "YOLO11 Large  ~49 MB  accurate, ~50 FPS GPU  ← best for RTX 3050"),
    "yolo11x": ("yolo11x.pt", "YOLO11 XLarge ~110 MB highest accuracy, ~30 FPS GPU"),
    # YOLOv8 — proven, widely used
    "yolov8n": ("yolov8n.pt", "YOLOv8 Nano   ~6 MB"),
    "yolov8s": ("yolov8s.pt", "YOLOv8 Small  ~22 MB"),
    "yolov8m": ("yolov8m.pt", "YOLOv8 Medium ~52 MB"),
    "yolov8l": ("yolov8l.pt", "YOLOv8 Large  ~87 MB"),
    "yolov8x": ("yolov8x.pt", "YOLOv8 XLarge ~131 MB"),
}

# Recommended for RTX 3050: YOLO11l balances accuracy + speed well
DEFAULT_MODEL    = "yolo11l"
MODELS_DIR       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class TrafficAnalyzer:
    """
    Main analysis engine.

    Parameters
    ----------
    model_key           : Key from AVAILABLE_MODELS. Default 'yolo11l'.
    use_tensorrt        : Export to TensorRT FP16 for maximum GPU speed.
                          Engine is compiled once and cached.
    use_fp16            : Use FP16 precision (PyTorch half). Default True on GPU.
    confidence          : Detection confidence threshold.
    iou_threshold       : NMS IoU threshold for tracking.
    pixels_per_meter    : Speed calibration (px/m). Default 20.
    max_density         : Vehicle count = congestion score 100.
    counting_line_y_frac: Y position of counting line (0–1 fraction of height).
    progress_cb         : Optional callback(downloaded_bytes, total_bytes).
    """

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        use_tensorrt: bool = False,
        use_fp16: bool = True,
        confidence: float = 0.45,
        iou_threshold: float = 0.5,
        pixels_per_meter: float = 20.0,
        max_density: int = 20,
        counting_line_y_frac: float = 0.55,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        # Legacy compat params (ignored if ultralytics available)
        model_path: str = None,
        config_path: str = None,
        names_path: str = None,
        nms_threshold: float = 0.4,
    ):
        self.model_key            = model_key if model_key in AVAILABLE_MODELS else DEFAULT_MODEL
        self.use_tensorrt         = use_tensorrt
        self.use_fp16             = use_fp16
        self.confidence           = confidence
        self.iou_threshold        = iou_threshold
        self.pixels_per_meter     = pixels_per_meter
        self.max_density          = max_density
        self.counting_line_y_frac = counting_line_y_frac
        self.progress_cb          = progress_cb

        self.counting_line_y: Optional[int] = None
        self.frame_count   = 0
        self.total_count   = 0
        self.vehicle_type_counts: dict[str, int] = defaultdict(int)
        self.counted_ids: set[int] = set()
        self.density_history: deque = deque(maxlen=150)
        self.speed_estimates: deque = deque(maxlen=100)
        self.fps_history: deque    = deque(maxlen=30)
        self._prev_centroids: dict[int, np.ndarray] = {}
        self._prev_time = time.time()
        self._id_to_class: dict[int, str] = {}

        self._use_ultralytics = False
        self._model = None
        self._device = "cpu"

        # Legacy YOLOv3 fallback
        self._legacy_net     = None
        self._legacy_layers  = None

        os.makedirs(MODELS_DIR, exist_ok=True)
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        # ── Step 1: check ultralytics + torch are importable ─────────────────
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:
            logger.warning(
                "ultralytics or PyTorch not installed — falling back to YOLOv3.\n"
                "  Install:  pip install ultralytics torch torchvision"
            )
            self._load_legacy_yolov3()
            return

        # ── Step 2: detect GPU ────────────────────────────────────────────────
        if torch.cuda.is_available():
            self._device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU detected: %s  (%.1f GB VRAM)", gpu_name, vram_gb)
        else:
            self._device = "cpu"
            self.use_tensorrt = False   # TensorRT requires GPU
            logger.info("No CUDA GPU found — using CPU.")

        model_file, desc = AVAILABLE_MODELS[self.model_key]
        model_pt    = os.path.join(MODELS_DIR, model_file)
        engine_file = os.path.join(MODELS_DIR, model_file.replace(".pt", "_fp16.engine"))

        # ── Step 3: try TensorRT path (optional, gracefully degrades) ─────────
        if self.use_tensorrt and self._device != "cpu":
            trt_ok = self._try_load_tensorrt(YOLO, model_pt, engine_file)
            if trt_ok:
                return
            # TensorRT failed (e.g. tensorrt not installed on Windows)
            # → fall through to normal PyTorch loading, do NOT go to YOLOv3
            logger.warning(
                "TensorRT unavailable — loading %s in PyTorch FP16 mode instead.\n"
                "  To enable TensorRT on Windows, install NVIDIA TensorRT:\n"
                "  https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html\n"
                "  Or use ONNX runtime:  pip install onnxruntime-gpu",
                model_file,
            )
            self.use_tensorrt = False

        # ── Step 4: normal PyTorch loading ────────────────────────────────────
        try:
            logger.info("Loading %s (%s) …", self.model_key, desc)
            self._model = YOLO(model_pt)

            if self.use_fp16 and self._device != "cpu":
                self._model.model.half()
                logger.info("FP16 (half precision) enabled.")

            self._use_ultralytics = True
            logger.info(
                "TrafficAnalyzer ready — %s | device=%s | FP16=%s",
                self.model_key, self._device,
                "yes" if (self.use_fp16 and self._device != "cpu") else "no",
            )
        except Exception as ex:
            logger.error("Failed to load %s: %s", model_file, ex)
            logger.warning("Falling back to YOLOv3 cv2.dnn.")
            self._load_legacy_yolov3()

    def _try_load_tensorrt(self, YOLO, model_pt: str, engine_file: str) -> bool:
        """
        Attempt to compile/load a TensorRT engine.
        Returns True on success, False on any failure.
        TensorRT requires:
          - NVIDIA TensorRT SDK (not pip-installable on Windows directly)
          - Windows: install via https://developer.nvidia.com/tensorrt
          - Linux:   pip install tensorrt  (works on Linux with CUDA)
        """
        try:
            # Cached engine — just load it
            if os.path.exists(engine_file):
                logger.info("Loading cached TensorRT engine: %s", engine_file)
                self._model = YOLO(engine_file)
                self._use_ultralytics = True
                logger.info("TensorRT FP16 engine loaded — maximum GPU speed active.")
                return True

            # No cache — compile it
            logger.info("Compiling TensorRT FP16 engine for %s…", model_pt)
            logger.info("This takes 1–3 minutes and is cached permanently after.")
            base = YOLO(model_pt)
            base.export(
                format="engine",
                half=True,
                device=self._device,
                imgsz=640,
            )
            # Ultralytics saves the engine next to the .pt file
            exported = model_pt.replace(".pt", ".engine")
            if os.path.exists(exported):
                os.rename(exported, engine_file)
                logger.info("TensorRT engine cached: %s", engine_file)
            elif not os.path.exists(engine_file):
                raise FileNotFoundError("Engine file not found after export.")

            self._model = YOLO(engine_file)
            self._use_ultralytics = True
            logger.info("TensorRT FP16 engine loaded — maximum GPU speed active.")
            return True

        except Exception as ex:
            logger.warning("TensorRT export/load failed: %s", ex)
            # Clean up any partial .onnx or .engine files left by failed export
            for ext in (".onnx", ".engine"):
                leftover = model_pt.replace(".pt", ext)
                if os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except Exception:
                        pass
            return False

    def _load_legacy_yolov3(self):
        """YOLOv3 via cv2.dnn — only used when ultralytics is unavailable."""
        import urllib.request
        weights = os.path.join(MODELS_DIR, "yolov3.weights")
        cfg     = os.path.join(MODELS_DIR, "yolov3.cfg")
        names   = os.path.join(MODELS_DIR, "coco.names")

        if not os.path.exists(cfg):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
                cfg,
            )
        if not os.path.exists(names):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                names,
            )

        if not os.path.exists(weights):
            mirrors = [
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3.pt",
                "https://pjreddie.com/media/files/yolov3.weights",
            ]
            downloaded = False
            for url in mirrors:
                try:
                    def _hook(b, bs, total, _url=url):
                        done = b * bs
                        if self.progress_cb and total > 0:
                            self.progress_cb(done, total)
                        if total > 0 and b % 300 == 0:
                            logger.info("YOLOv3 download %d%% from %s",
                                        min(100, done * 100 // total), _url)
                    urllib.request.urlretrieve(url, weights, reporthook=_hook)
                    downloaded = True
                    break
                except Exception as e:
                    logger.warning("Mirror %s failed: %s", url, e)
                    if os.path.exists(weights):
                        os.remove(weights)
            if not downloaded:
                raise RuntimeError(
                    "Cannot download YOLOv3 weights.\n"
                    "Run:  pip install ultralytics torch torchvision"
                )

        net = cv2.dnn.readNet(weights, cfg)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("YOLOv3 legacy: CUDA enabled.")
        layer_names = net.getLayerNames()
        self._legacy_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        self._legacy_net    = net
        self.model_key      = "yolov3-legacy"
        logger.info("YOLOv3 legacy (cv2.dnn) loaded.")

    # ── Frame processing ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        t0 = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]

        if self.counting_line_y is None:
            self.counting_line_y = int(h * self.counting_line_y_frac)

        if self._use_ultralytics:
            annotated, tracked_objs = self._run_ultralytics(frame)
        else:
            annotated, tracked_objs = self._run_legacy(frame)

        # Counting + speed estimation
        now = time.time()
        dt  = max(now - self._prev_time, 1e-6)

        for obj_id, centroid, vtype in tracked_objs:
            if obj_id not in self._id_to_class:
                self._id_to_class[obj_id] = vtype
            if obj_id not in self.counted_ids and centroid[1] >= self.counting_line_y:
                self.counted_ids.add(obj_id)
                self.total_count += 1
                self.vehicle_type_counts[self._id_to_class[obj_id]] += 1
            prev = self._prev_centroids.get(obj_id)
            if prev is not None:
                pixel_dist = float(np.linalg.norm(centroid - prev))
                speed_kmh  = (pixel_dist / self.pixels_per_meter / dt) * 3.6
                if 0.5 < speed_kmh < 250:
                    self.speed_estimates.append(speed_kmh)

        self._prev_centroids = {oid: c for oid, c, _ in tracked_objs}
        self._prev_time = now

        density = len(tracked_objs)
        self.density_history.append(density)
        avg_density      = float(np.mean(self.density_history))
        congestion_score = min(100, int(avg_density * 100 / max(1, self.max_density)))
        congestion_level = self._get_level(congestion_score)
        avg_speed        = float(np.mean(self.speed_estimates)) if self.speed_estimates else 0.0

        fps = 1.0 / max(time.time() - t0, 1e-6)
        self.fps_history.append(fps)

        self._draw_overlay(annotated, congestion_level, congestion_score, avg_speed, density)

        return {
            "frame":               annotated,
            "total_count":         self.total_count,
            "current_density":     density,
            "congestion_score":    congestion_score,
            "congestion_level":    congestion_level,
            "avg_speed_kmh":       round(avg_speed, 1),
            "vehicle_type_counts": dict(self.vehicle_type_counts),
            "fps":                 round(float(np.mean(self.fps_history)), 1),
            "timestamp":           datetime.now(),
            "model":               self.model_key,
        }

    # ── Ultralytics YOLO11/YOLOv8 inference ──────────────────────────────────

    def _run_ultralytics(self, frame: np.ndarray):
        """
        Run detection + ByteTracker in one call.
        Returns (annotated_frame, [(track_id, centroid, vtype), ...])
        """
        results = self._model.track(
            frame,
            persist=True,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=list(VEHICLE_CLASS_IDS.keys()),
            device=self._device,
            half=(self.use_fp16 and self._device != "cpu" and not self.use_tensorrt),
            verbose=False,
            imgsz=640,
            tracker="bytetrack.yaml",  # ByteTracker — best for traffic
        )
        result   = results[0]
        annotated = result.plot(
            line_width=2,
            font_size=0.5,
        )

        tracked_objs = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy     = boxes.xyxy.cpu().numpy()
            cls_ids  = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)
            for box, cls_id, tid in zip(xyxy, cls_ids, track_ids):
                vtype = VEHICLE_CLASS_IDS.get(cls_id)
                if vtype is None:
                    continue
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                tracked_objs.append((int(tid), np.array([cx, cy]), vtype))

        return annotated, tracked_objs

    # ── Legacy YOLOv3 (cv2.dnn) ───────────────────────────────────────────────

    def _run_legacy(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self._legacy_net.setInput(blob)
        outputs = self._legacy_net.forward(self._legacy_layers)

        boxes, confs, cls_ids = [], [], []
        for out in outputs:
            for det in out:
                scores  = det[5:]
                cid     = int(np.argmax(scores))
                conf    = float(scores[cid])
                if conf < self.confidence or cid not in VEHICLE_CLASS_IDS:
                    continue
                cx = int(det[0]*w); cy = int(det[1]*h)
                bw = int(det[2]*w); bh = int(det[3]*h)
                boxes.append([cx-bw//2, cy-bh//2, bw, bh])
                confs.append(conf)
                cls_ids.append(cid)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.confidence, self.iou_threshold)
        annotated    = frame.copy()
        tracked_objs = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, bw, bh = boxes[i]
                vtype = VEHICLE_CLASS_IDS.get(cls_ids[i], "car")
                cx = x + bw // 2
                cy = y + bh // 2
                cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (50,230,50), 2)
                cv2.putText(annotated, vtype, (x, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,230,50), 1)
                pseudo_id = hash((round(cx/40), round(cy/40))) & 0x7FFF
                tracked_objs.append((pseudo_id, np.array([cx, cy]), vtype))
        return annotated, tracked_objs

    # ── Overlay: counting line + stats HUD ───────────────────────────────────

    def _draw_overlay(self, frame, level, score, speed, density):
        h, w = frame.shape[:2]
        level_colors = {
            "low":      (0, 200, 100),
            "moderate": (0, 180, 255),
            "high":     (0, 100, 255),
            "severe":   (0,   0, 220),
        }
        c = level_colors[level]

        # Counting line
        cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y),
                 (50, 230, 50), 2)
        cv2.putText(frame, "COUNTING LINE", (8, self.counting_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 230, 50), 1)

        # HUD background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 120), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # HUD text lines
        lines = [
            (f"Model: {self.model_key}", (160, 160, 160)),
            (f"Count: {self.total_count}   Live: {density}", (240, 240, 240)),
            (f"Congestion: {level.upper()} ({score}/100)", c),
            (f"Avg speed: {speed:.1f} km/h", (200, 200, 200)),
            (f"FPS: {round(float(np.mean(self.fps_history)) if self.fps_history else 0, 1)}", (140, 140, 140)),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (8, 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_level(self, score: int) -> str:
        if score >= 80: return "severe"
        if score >= 60: return "high"
        if score >= 30: return "moderate"
        return "low"

    def reset_counts(self):
        self.vehicle_type_counts = defaultdict(int)
        self.total_count   = 0
        self.frame_count   = 0
        self.counted_ids   = set()
        self._id_to_class  = {}
        self._prev_centroids = {}
        self.density_history.clear()
        self.speed_estimates.clear()
        self.fps_history.clear()
        self.counting_line_y = None
        logger.info("Counts reset.")

    def set_counting_line(self, y_frac: float):
        self.counting_line_y_frac = max(0.1, min(0.95, y_frac))
        self.counting_line_y = None

    @staticmethod
    def list_models() -> list[tuple[str, str]]:
        """Return [(key, description), ...] for the model selector UI."""
        return [(k, v[1]) for k, v in AVAILABLE_MODELS.items()]
