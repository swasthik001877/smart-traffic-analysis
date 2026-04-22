"""
Core traffic analyzer — YOLO-based vehicle detection, tracking, counting,
congestion scoring, and route suggestion.
"""

import cv2
import numpy as np
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)

# YOLO COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    1:  "bicycle",
}

CONGESTION_THRESHOLDS = {
    "low":      (0,  30),
    "moderate": (30, 60),
    "high":     (60, 80),
    "severe":   (80, 100),
}

ROUTE_SUGGESTIONS = {
    "low":      ("Current route is optimal.", "green"),
    "moderate": ("Slight delays ahead. Consider Ring Road as alternative.", "yellow"),
    "high":     ("Heavy congestion. Recommended: Ring Road → Connector B → SH-12 South.", "orange"),
    "severe":   ("Severe jam. Use: Ring Road North → City Bypass (+4 min, 58 km/h avg).", "red"),
}


class VehicleTracker:
    """Simple centroid-based vehicle tracker using OpenCV."""

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict[int, np.ndarray] = {}
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.counted_ids: set[int] = set()

    def register(self, centroid: np.ndarray) -> int:
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, centroids: list[np.ndarray]) -> dict[int, np.ndarray]:
        if not centroids:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        if not self.objects:
            for c in centroids:
                self.register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())
        D = np.linalg.norm(
            np.array(obj_centroids)[:, np.newaxis] - np.array(centroids)[np.newaxis, :],
            axis=2
        )
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > 80:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols
        for row in unused_rows:
            self.disappeared[obj_ids[row]] += 1
            if self.disappeared[obj_ids[row]] > self.max_disappeared:
                self.deregister(obj_ids[row])
        for col in unused_cols:
            self.register(centroids[col])

        return self.objects


class TrafficAnalyzer:
    """
    Main analysis engine.
    Loads YOLO, processes frames, tracks vehicles, computes congestion,
    and provides route recommendations.
    """

    def __init__(
        self,
        confidence: float = 0.5,
        nms_threshold: float = 0.4,
        model_path: str = "models/yolov3.weights",
        config_path: str = "models/yolov3.cfg",
        names_path: str = "models/coco.names",
    ):
        self.confidence_threshold = confidence
        self.nms_threshold = nms_threshold
        self.tracker = VehicleTracker()
        self.vehicle_type_counts: dict[str, int] = defaultdict(int)
        self.total_count = 0
        self.frame_count = 0
        self.fps_history: deque = deque(maxlen=30)
        self.density_history: deque = deque(maxlen=150)
        self.speed_estimates: deque = deque(maxlen=100)
        self._prev_centroids: dict[int, np.ndarray] = {}
        self._prev_time = time.time()
        self.counting_line_y: Optional[int] = None

        self.net, self.output_layers, self.class_names = self._load_model(
            model_path, config_path, names_path
        )
        logger.info("TrafficAnalyzer initialized.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self, weights, cfg, names):
        import os
        if not os.path.exists(weights):
            logger.warning(f"YOLO weights not found at {weights}. Downloading...")
            self._download_yolo()

        net = cv2.dnn.readNet(weights, cfg)
        # NEW — use CUDA if available, fall back to CPU
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("GPU (CUDA) backend enabled.")
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("CUDA not available — using CPU.")

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        with open(names, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        logger.info(f"YOLO model loaded — {len(class_names)} classes")
        return net, output_layers, class_names

    def _download_yolo(self):
        """Download YOLOv3 weights, cfg, and coco.names if missing."""
        import os, urllib.request
        os.makedirs("models", exist_ok=True)
        files = {
            "models/yolov3.cfg":
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "models/coco.names":
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        }
        for path, url in files.items():
            if not os.path.exists(path):
                logger.info(f"Downloading {path}...")
                urllib.request.urlretrieve(url, path)
        weights_path = "models/yolov3.weights"
        if not os.path.exists(weights_path):
            logger.info("Downloading YOLOv3 weights (236 MB)...")
            urllib.request.urlretrieve(
                "https://pjreddie.com/media/files/yolov3.weights", weights_path
            )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run detection + tracking on a single frame.
        Returns: dict with annotated frame and all stats.
        """
        t0 = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]

        if self.counting_line_y is None:
            self.counting_line_y = int(h * 0.55)

        # --- YOLO detection ---
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf < self.confidence_threshold or class_id not in VEHICLE_CLASSES:
                    continue
                cx, cy, bw, bh = (
                    int(detection[0] * w), int(detection[1] * h),
                    int(detection[2] * w), int(detection[3] * h),
                )
                x = cx - bw // 2
                y = cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        centroids, detected_classes, detected_boxes = [], [], []

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, bw, bh = boxes[i]
                cx = x + bw // 2
                cy = y + bh // 2
                centroids.append(np.array([cx, cy]))
                detected_classes.append(class_ids[i])
                detected_boxes.append((x, y, bw, bh, confidences[i]))

        # --- Tracking ---
        tracked = self.tracker.update(centroids)

        # --- Counting line logic ---
        now = time.time()
        dt = now - self._prev_time
        for obj_id, centroid in tracked.items():
            if obj_id in self.tracker.counted_ids:
                continue
            if centroid[1] >= self.counting_line_y:
                self.tracker.counted_ids.add(obj_id)
                self.total_count += 1
                # Assign vehicle type
                if centroids:
                    dists = [np.linalg.norm(centroid - c) for c in centroids]
                    nearest_idx = int(np.argmin(dists))
                    if nearest_idx < len(detected_classes):
                        vtype = VEHICLE_CLASSES.get(detected_classes[nearest_idx], "car")
                        self.vehicle_type_counts[vtype] += 1

        # --- Speed estimation (pixel displacement / time → km/h proxy) ---
        for obj_id, centroid in tracked.items():
            if obj_id in self._prev_centroids and dt > 0:
                prev = self._prev_centroids[obj_id]
                pixel_dist = np.linalg.norm(centroid - prev)
                # Rough scale: 100px ≈ 5m at typical camera height
                speed_kmh = (pixel_dist / 100 * 5) / dt * 3.6
                if 0 < speed_kmh < 150:
                    self.speed_estimates.append(speed_kmh)

        self._prev_centroids = {k: v.copy() for k, v in tracked.items()}
        self._prev_time = now

        # --- Density & congestion ---
        density = len(tracked)
        self.density_history.append(density)
        avg_density = float(np.mean(self.density_history))
        congestion_score = min(100, int(avg_density * 100 / 15))
        congestion_level = self._get_congestion_level(congestion_score)
        avg_speed = float(np.mean(self.speed_estimates)) if self.speed_estimates else 0.0

        # --- FPS ---
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.fps_history.append(fps)

        # --- Annotate frame ---
        annotated = self._annotate(
            frame.copy(), tracked, detected_boxes, centroids,
            congestion_level, congestion_score, avg_speed
        )

        return {
            "frame": annotated,
            "total_count": self.total_count,
            "current_density": density,
            "congestion_score": congestion_score,
            "congestion_level": congestion_level,
            "avg_speed_kmh": round(avg_speed, 1),
            "vehicle_type_counts": dict(self.vehicle_type_counts),
            "fps": round(float(np.mean(self.fps_history)), 1),
            "route_suggestion": ROUTE_SUGGESTIONS[congestion_level],
            "timestamp": datetime.now(),
        }

    def _get_congestion_level(self, score: int) -> str:
        if score >= 80:
            return "severe"
        elif score >= 60:
            return "high"
        elif score >= 30:
            return "moderate"
        else:
            return "low"

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------
    def _annotate(self, frame, tracked, detected_boxes, centroids,
                  congestion_level, congestion_score, avg_speed):
        h, w = frame.shape[:2]
        colors = {
            "low":      (0, 200, 100),
            "moderate": (0, 180, 255),
            "high":     (0, 100, 255),
            "severe":   (0, 0, 220),
        }
        color = colors[congestion_level]

        # Counting line
        cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y), (50, 230, 50), 2)
        cv2.putText(frame, "COUNTING LINE", (8, self.counting_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 230, 50), 1)

        # Bounding boxes
        for x, y, bw, bh, conf in detected_boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        # Tracker IDs
        for obj_id, centroid in tracked.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(frame, f"#{obj_id}", (cx + 6, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

        # HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        route_msg, _ = ROUTE_SUGGESTIONS[congestion_level]
        lines = [
            (f"Count: {self.total_count}  |  Live: {len(tracked)}", (255, 255, 255)),
            (f"Congestion: {congestion_level.upper()} ({congestion_score}/100)", color),
            (f"Avg speed: {avg_speed:.1f} km/h", (200, 200, 200)),
            (f"FPS: {round(float(np.mean(self.fps_history)), 1)}", (160, 160, 160)),
            (f"Route: {route_msg[:42]}", (100, 220, 255)),
        ]
        for i, (text, c) in enumerate(lines):
            cv2.putText(frame, text, (8, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1, cv2.LINE_AA)

        return frame

    def reset_counts(self):
        self.vehicle_type_counts = defaultdict(int)
        self.total_count = 0
        self.frame_count = 0
        self.tracker = VehicleTracker()
        self.density_history.clear()
        logger.info("Counts reset.")
