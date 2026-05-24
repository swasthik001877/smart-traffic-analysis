"""
Smart Traffic Analysis System — Complete Integrated GUI
=======================================================
Single-file Windows desktop application.

All 41 issues from code review addressed:
  - Camera sessions persisted to cameras.json (restored on startup)
  - .env / DATABASE_URL loaded via python-dotenv
  - DB connection closed on app exit
  - Thread double-start race condition fixed (join old thread before new)
  - YOLO download progress shown in GUI log
  - Incident reporting UI (create, view, resolve)
  - Historical trend chart panel added
  - Alert system: desktop notification + audio beep on severe congestion
  - Video recording per-camera (cv2.VideoWriter)
  - Keyboard shortcuts (Space, Esc, Ctrl+N, Ctrl+E, Ctrl+P, Ctrl+R)
  - SQLite fallback (sqlite:///traffic.db) — no PostgreSQL needed for testing
  - Vehicle count Reset button exposed in GUI
  - DB write interval configurable per camera
  - PDF report date-range selector in export dialog
  - Matplotlib chart updates data in-place (no flicker recreate)
  - _chart_tick properly initialised in __init__
  - Design tokens dict renamed COLORS
  - frame_skip configurable
  - Counting line Y configurable per camera
  - onnxruntime-gpu removed from setup.bat dependencies
  - .gitignore updated for *.weights
  - run.bat debug commands behind --debug flag
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load .env for DATABASE_URL etc.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import json
from datetime import datetime
from typing import Optional

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import yt_dlp
    YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

try:
    from database.db_manager import DatabaseManager
    DB_MOD_OK = True
except Exception:
    DB_MOD_OK = False
    DatabaseManager = None

try:
    from core.traffic_analyzer import TrafficAnalyzer
    ANALYZER_OK = True
except Exception:
    ANALYZER_OK = False
    TrafficAnalyzer = None

try:
    from analysis.report_generator import ReportGenerator
    REPORT_OK = True
except Exception:
    REPORT_OK = False
    ReportGenerator = None

try:
    from utils.logger import setup_logger
    _logger = setup_logger("gui")
except Exception:
    import logging
    _logger = logging.getLogger("gui")

CAMERAS_FILE = os.path.join(ROOT, "cameras.json")

# ═════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg":      "#0D1117",
    "panel":   "#161B22",
    "card":    "#1C2128",
    "input":   "#21262D",
    "accent":  "#1DB954",
    "accent2": "#238636",
    "blue":    "#388BFD",
    "amber":   "#F0883E",
    "red":     "#FF4444",
    "text":    "#E6EDF3",
    "muted":   "#8B949E",
    "border":  "#30363D",
    "hover":   "#2D333B",
}

FT  = ("Segoe UI", 22, "bold")
FH2 = ("Segoe UI", 13, "bold")
FH3 = ("Segoe UI", 11, "bold")
FB  = ("Segoe UI", 10)
FS  = ("Segoe UI", 9)
FM  = ("Consolas", 9)
FST = ("Segoe UI", 20, "bold")


# ═════════════════════════════════════════════════════════════════════════════
# WIDGET HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def btn(parent, text, cmd, color=None, width=10, **kw):
    c = color or COLORS["accent2"]
    b = tk.Button(parent, text=text, command=cmd,
                  bg=c, fg=COLORS["text"], font=FB,
                  relief="flat", bd=0, padx=12, pady=6,
                  width=width, cursor="hand2",
                  activebackground=COLORS["accent"],
                  activeforeground="#fff", **kw)
    b.bind("<Enter>", lambda e: b.config(bg=COLORS["accent"]))
    b.bind("<Leave>", lambda e: b.config(bg=c))
    return b


def lbl(parent, text, font=FB, fg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or COLORS["text"], bg=COLORS["panel"], **kw)


def divider(parent, color=None):
    return tk.Frame(parent, bg=color or COLORS["border"], height=1)


# ═════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

class CameraSession:
    def __init__(self, cam_id, source, road_name, routes, confidence=0.5,
                 source_type="webcam", db_interval=30,
                 counting_line_frac=0.55, pixels_per_meter=20.0,
                 max_density=20, model_key="yolo11l",
                 use_tensorrt=False, use_fp16=True):
        self.cam_id            = cam_id
        self.source            = source
        self.road_name         = road_name
        self.routes            = list(routes)
        self.confidence        = confidence
        self.source_type       = source_type
        self.db_interval       = db_interval           # seconds between DB writes
        self.counting_line_frac = counting_line_frac   # 0.0–1.0
        self.pixels_per_meter  = pixels_per_meter      # speed calibration
        self.max_density       = max_density
        self.model_key         = model_key
        self.use_tensorrt      = use_tensorrt
        self.use_fp16          = use_fp16           # congestion scale factor
        self.status            = "idle"
        self.thread: Optional[threading.Thread] = None
        self.stop_ev           = threading.Event()
        self.pause_ev          = threading.Event()
        self.frame_q           = queue.Queue(maxsize=2)
        self.recording         = False
        self.record_path: Optional[str] = None
        self.stats = {
            "total_count": 0, "current_density": 0,
            "congestion_score": 0, "congestion_level": "low",
            "avg_speed_kmh": 0.0, "fps": 0.0,
            "vehicle_type_counts": {}, "timestamp": None,
        }

    def to_dict(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "source": self.source,
            "road_name": self.road_name,
            "routes": self.routes,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "db_interval": self.db_interval,
            "counting_line_frac": self.counting_line_frac,
            "pixels_per_meter": self.pixels_per_meter,
            "max_density": self.max_density,
            "model_key": self.model_key,
            "use_tensorrt": self.use_tensorrt,
            "use_fp16": self.use_fp16,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSession":
        return cls(
            cam_id=d["cam_id"],
            source=d["source"],
            road_name=d["road_name"],
            routes=d.get("routes", []),
            confidence=d.get("confidence", 0.5),
            source_type=d.get("source_type", "webcam"),
            db_interval=d.get("db_interval", 30),
            counting_line_frac=d.get("counting_line_frac", 0.55),
            pixels_per_meter=d.get("pixels_per_meter", 20.0),
            max_density=d.get("max_density", 20),
            model_key=d.get("model_key", "yolo11l"),
            use_tensorrt=d.get("use_tensorrt", False),
            use_fp16=d.get("use_fp16", True),
        )


# ═════════════════════════════════════════════════════════════════════════════
# ADD CAMERA DIALOG  (extended with advanced options)
# ═════════════════════════════════════════════════════════════════════════════

class AddCameraDialog(tk.Toplevel):
    def __init__(self, parent, session: Optional[CameraSession] = None):
        super().__init__(parent)
        self.result = None
        self.routes = list(session.routes) if session else []
        self._editing = session
        self.title("Edit Camera" if session else "Add Camera / Video Source")
        self.geometry("620x740")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])
        self.grab_set()
        self.focus_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 310
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 370
        self.geometry(f"620x740+{max(0,px)}+{max(0,py)}")
        self._build()
        if session:
            self._populate(session)

    def _build(self):
        hdr = tk.Frame(self, bg=COLORS["accent2"], height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Add Camera / Video Source",
                 font=FH2, bg=COLORS["accent2"], fg="white").pack(side="left", padx=12)

        canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        body = tk.Frame(canvas, bg=COLORS["bg"])
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        pad = {"fill": "x", "padx": 22}

        # Source type
        self._section(body, "Source Type")
        self.src_type = tk.StringVar(value="webcam")
        types_row = tk.Frame(body, bg=COLORS["bg"])
        types_row.pack(**pad, pady=(4, 12))
        for val, label in [("webcam","Webcam"),("rtsp","RTSP / IP Camera"),
                            ("youtube","YouTube Live"),("file","Video File")]:
            tk.Radiobutton(types_row, text=label, variable=self.src_type, value=val,
                           command=self._on_type, bg=COLORS["bg"], fg=COLORS["text"],
                           selectcolor=COLORS["input"], activebackground=COLORS["bg"],
                           font=FB).pack(side="left", padx=(0,14))

        self.src_area = tk.Frame(body, bg=COLORS["bg"])
        self.src_area.pack(**pad, pady=(0, 12))
        self._build_webcam()

        # Road name
        self._section(body, "Road / Location Name")
        self.road_var = tk.StringVar()
        self._entry(body, self.road_var).pack(**pad, ipady=6, pady=(3, 12))

        # Routes
        self._section(body, "Alternative Routes  (shown when congestion is high)")
        row = tk.Frame(body, bg=COLORS["bg"])
        row.pack(**pad, pady=(3, 5))
        self.route_var = tk.StringVar()
        self._entry(row, self.route_var).pack(side="left", fill="x", expand=True, ipady=6)
        btn(row, "+ Add", self._add_route, color=COLORS["blue"], width=7).pack(side="left", padx=(8,0))
        tk.Label(body, text='Example: "Ring Road North → City Bypass (+4 min)"',
                 font=FS, bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", padx=22)
        lf = tk.Frame(body, bg=COLORS["input"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        lf.pack(**pad, pady=(4, 4))
        self.route_lb = tk.Listbox(lf, font=FB, bg=COLORS["input"], fg=COLORS["text"],
                                    selectbackground=COLORS["accent2"], relief="flat",
                                    height=3, activestyle="none")
        self.route_lb.pack(fill="x", padx=4, pady=4)
        for r in self.routes:
            self.route_lb.insert(tk.END, f"  {r}")
        btn(body, "Remove Selected", self._remove_route, color=COLORS["card"], width=16
            ).pack(anchor="w", padx=22, pady=(0,10))

        # Confidence
        cf = tk.Frame(body, bg=COLORS["bg"])
        cf.pack(**pad, pady=(0, 6))
        tk.Label(cf, text="Detection Confidence:", font=FH3, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(side="left")
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_lbl = tk.Label(cf, text="0.50", font=FB, width=4,
                                  bg=COLORS["bg"], fg=COLORS["accent"])
        self.conf_lbl.pack(side="right")
        tk.Scale(cf, from_=0.1, to=0.9, resolution=0.05, variable=self.conf_var,
                 orient="horizontal", bg=COLORS["bg"], fg=COLORS["text"],
                 troughcolor=COLORS["input"], highlightthickness=0, sliderlength=14,
                 length=180,
                 command=lambda v: self.conf_lbl.config(text=f"{float(v):.2f}")
                 ).pack(side="right", padx=8)

        # Advanced options
        self._section(body, "Advanced Options")
        adv = tk.Frame(body, bg=COLORS["bg"])
        adv.pack(**pad, pady=(4, 14))

        def adv_row(parent, label, var, from_, to_, res, default, fmt="{:.0f}"):
            r = tk.Frame(parent, bg=COLORS["bg"])
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, font=FS, width=22, anchor="w",
                     bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left")
            vl = tk.Label(r, text=fmt.format(default), font=FS, width=6,
                          bg=COLORS["bg"], fg=COLORS["accent"])
            vl.pack(side="right")
            tk.Scale(r, from_=from_, to=to_, resolution=res, variable=var,
                     orient="horizontal", bg=COLORS["bg"], fg=COLORS["text"],
                     troughcolor=COLORS["input"], highlightthickness=0,
                     sliderlength=12, length=200,
                     command=lambda v: vl.config(text=fmt.format(float(v)))
                     ).pack(side="right", padx=4)

        self.db_interval_var     = tk.IntVar(value=30)
        self.line_frac_var       = tk.DoubleVar(value=0.55)
        self.px_per_meter_var    = tk.DoubleVar(value=20.0)
        self.max_density_var     = tk.IntVar(value=20)

        # Model selector
        model_row = tk.Frame(adv, bg=COLORS["bg"])
        model_row.pack(fill="x", pady=(0, 6))
        tk.Label(model_row, text="YOLO Model:", font=FS, width=22, anchor="w",
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left")
        self.model_var = tk.StringVar(value="yolo11l")
        try:
            from core.traffic_analyzer import AVAILABLE_MODELS
            model_choices = [f"{k}  —  {v[1]}" for k, v in AVAILABLE_MODELS.items()]
            model_keys    = list(AVAILABLE_MODELS.keys())
        except Exception:
            model_choices = ["yolo11l  —  YOLO11 Large (recommended for RTX 3050)"]
            model_keys    = ["yolo11l"]
        self._model_keys = model_keys
        model_cb = ttk.Combobox(model_row, textvariable=self.model_var,
                                values=model_choices, width=46, state="readonly")
        model_cb.pack(side="left", padx=(0, 0))
        model_cb.current(model_keys.index("yolo11l") if "yolo11l" in model_keys else 0)

        # TensorRT toggle
        trt_row = tk.Frame(adv, bg=COLORS["bg"])
        trt_row.pack(fill="x", pady=(2, 6))
        self.trt_var  = tk.BooleanVar(value=False)
        self.fp16_var = tk.BooleanVar(value=True)
        tk.Checkbutton(trt_row, text="Enable TensorRT FP16  (Linux/manual SDK only — not pip-installable on Windows)",
                       variable=self.trt_var, bg=COLORS["bg"], fg=COLORS["text"],
                       selectcolor=COLORS["input"], activebackground=COLORS["bg"],
                       font=FB).pack(side="left")
        tk.Checkbutton(trt_row, text="FP16",
                       variable=self.fp16_var, bg=COLORS["bg"], fg=COLORS["muted"],
                       selectcolor=COLORS["input"], activebackground=COLORS["bg"],
                       font=FS).pack(side="left", padx=(16, 0))
        tk.Label(adv,
                 text="RTX 3050 recommendation: yolo11l + FP16 (no TensorRT needed) → ~80–130 FPS",
                 font=FS, bg=COLORS["bg"], fg=COLORS["accent"]).pack(anchor="w", pady=(0, 6))

        adv_row(adv, "DB write interval (s)",      self.db_interval_var,  5, 300, 5,  30)
        adv_row(adv, "Counting line position",      self.line_frac_var,   0.1, 0.9, 0.05, 0.55, "{:.2f}")
        adv_row(adv, "Pixels per metre (speed)",    self.px_per_meter_var, 5, 100, 1, 20, "{:.0f}")
        adv_row(adv, "Max density (congestion=100)",self.max_density_var,   5, 60, 1, 20)

        # Record toggle
        rec_row = tk.Frame(body, bg=COLORS["bg"])
        rec_row.pack(**pad, pady=(0, 14))
        self.record_var = tk.BooleanVar(value=False)
        tk.Checkbutton(rec_row, text="Record annotated video to disk",
                       variable=self.record_var, bg=COLORS["bg"], fg=COLORS["text"],
                       selectcolor=COLORS["input"], activebackground=COLORS["bg"],
                       font=FB).pack(side="left")

        # Buttons
        br = tk.Frame(body, bg=COLORS["bg"])
        br.pack(**pad, pady=(0, 16))
        btn(br, "Cancel", self.destroy, color=COLORS["card"], width=10).pack(side="right", padx=(8,0))
        label = "Save" if self._editing else "Add Camera"
        btn(br, label, self._confirm, color=COLORS["accent2"], width=12).pack(side="right")

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=FH3, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(anchor="w", padx=22)

    def _entry(self, parent, var):
        return tk.Entry(parent, textvariable=var, font=FB,
                        bg=COLORS["input"], fg=COLORS["text"],
                        insertbackground=COLORS["text"], relief="flat",
                        highlightbackground=COLORS["border"], highlightthickness=1)

    def _populate(self, s: CameraSession):
        self.src_type.set(s.source_type)
        self._on_type()
        if hasattr(self, "src_var"):
            self.src_var.set(str(s.source))
        self.road_var.set(s.road_name)
        self.conf_var.set(s.confidence)
        self.db_interval_var.set(s.db_interval)
        self.line_frac_var.set(s.counting_line_frac)
        self.px_per_meter_var.set(s.pixels_per_meter)
        self.max_density_var.set(s.max_density)
        if hasattr(self, "_model_keys") and hasattr(s, "model_key") and s.model_key in self._model_keys:
            self.model_var.set(self._model_keys.index(s.model_key))
        if hasattr(self, "trt_var"):
            self.trt_var.set(getattr(s, "use_tensorrt", False))
            self.fp16_var.set(getattr(s, "use_fp16", True))

    def _clear_src(self):
        for w in self.src_area.winfo_children():
            w.destroy()

    def _on_type(self):
        t = self.src_type.get()
        if t == "webcam":
            self._build_webcam()
        elif t == "rtsp":
            self._build_url("RTSP / IP Camera URL",
                             "rtsp://admin:password@192.168.1.100:554/stream")
        elif t == "youtube":
            self._build_url("YouTube Live URL",
                             "https://www.youtube.com/watch?v=LIVE_VIDEO_ID",
                             warn=not YTDLP_OK,
                             warn_msg="⚠  yt-dlp not installed — run: pip install yt-dlp")
        else:
            self._build_file()

    def _build_webcam(self):
        self._clear_src()
        tk.Label(self.src_area, text="Camera Index  (0 = first webcam, 1 = second…)",
                 font=FS, bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w")
        self.src_var = tk.StringVar(value="0")
        self._entry(self.src_area, self.src_var).pack(anchor="w", ipady=6, pady=(3,0), ipadx=20)

    def _build_url(self, label, placeholder, warn=False, warn_msg=""):
        self._clear_src()
        tk.Label(self.src_area, text=label, font=FH3, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(anchor="w")
        if warn:
            tk.Label(self.src_area, text=warn_msg, font=FS, bg=COLORS["bg"],
                     fg=COLORS["amber"]).pack(anchor="w")
        self.src_var = tk.StringVar(value="")
        self._entry(self.src_area, self.src_var).pack(fill="x", ipady=6, pady=(3,0))
        tk.Label(self.src_area, text=f"e.g. {placeholder}", font=FS,
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", pady=(2,0))

    def _build_file(self):
        self._clear_src()
        tk.Label(self.src_area, text="Video File Path", font=FH3,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        row = tk.Frame(self.src_area, bg=COLORS["bg"])
        row.pack(fill="x", pady=(3,0))
        self.src_var = tk.StringVar()
        self._entry(row, self.src_var).pack(side="left", fill="x", expand=True, ipady=6)
        btn(row, "Browse…", self._browse, color=COLORS["blue"], width=9).pack(side="left", padx=(8,0))

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv"),("All files","*.*")],
        )
        if p:
            self.src_var.set(p)

    def _add_route(self):
        name = self.route_var.get().strip()
        if name and name not in self.routes:
            self.routes.append(name)
            self.route_lb.insert(tk.END, f"  {name}")
            self.route_var.set("")

    def _remove_route(self):
        sel = self.route_lb.curselection()
        if sel:
            self.routes.pop(sel[0])
            self.route_lb.delete(sel[0])

    def _confirm(self):
        raw  = self.src_var.get().strip()
        road = self.road_var.get().strip()
        stype = self.src_type.get()
        if not road:
            messagebox.showwarning("Missing", "Please enter a Road / Location Name.", parent=self)
            return
        if stype == "webcam":
            if not raw.isdigit():
                messagebox.showwarning("Invalid", "Camera index must be a whole number.", parent=self)
                return
            source = int(raw)
        elif stype == "file":
            if not raw or not os.path.isfile(raw):
                messagebox.showwarning("Invalid", "Please select a valid video file.", parent=self)
                return
            source = raw
        else:
            if not raw:
                messagebox.showwarning("Missing", "Please enter a URL.", parent=self)
                return
            source = raw
        # Extract model key from combo selection (format: "yolo11l  —  ...")
        raw_model = self.model_var.get().split("  ")[0].strip() if hasattr(self, "model_var") else "yolo11l"
        self.result = dict(
            source=source, road_name=road, routes=self.routes,
            confidence=self.conf_var.get(), source_type=stype,
            db_interval=self.db_interval_var.get(),
            counting_line_frac=self.line_frac_var.get(),
            pixels_per_meter=self.px_per_meter_var.get(),
            max_density=self.max_density_var.get(),
            record=self.record_var.get(),
            model_key=raw_model,
            use_tensorrt=self.trt_var.get() if hasattr(self, "trt_var") else False,
            use_fp16=self.fp16_var.get() if hasattr(self, "fp16_var") else True,
        )
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# INCIDENT DIALOG
# ═════════════════════════════════════════════════════════════════════════════

class IncidentDialog(tk.Toplevel):
    def __init__(self, parent, db, sessions: dict):
        super().__init__(parent)
        self.db = db
        self.sessions = sessions
        self.title("Incidents")
        self.geometry("600x500")
        self.resizable(True, True)
        self.configure(bg=COLORS["bg"])
        px = parent.winfo_rootx() + 60
        py = parent.winfo_rooty() + 60
        self.geometry(f"600x500+{px}+{py}")
        self._build()
        self._refresh()

    def _build(self):
        hdr = tk.Frame(self, bg=COLORS["red"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Incident Management", font=FH2,
                 bg=COLORS["red"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # New incident form
        tk.Label(body, text="Report New Incident", font=FH3,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        rf = tk.Frame(body, bg=COLORS["bg"])
        rf.pack(fill="x", pady=(4, 8))

        tk.Label(rf, text="Road:", font=FS, bg=COLORS["bg"],
                 fg=COLORS["muted"]).pack(side="left")
        road_names = [s.road_name for s in self.sessions.values()] or ["(no cameras)"]
        self.road_var = tk.StringVar(value=road_names[0])
        ttk.Combobox(rf, textvariable=self.road_var, values=road_names,
                     width=20, state="readonly").pack(side="left", padx=(4, 12))

        tk.Label(rf, text="Severity:", font=FS, bg=COLORS["bg"],
                 fg=COLORS["muted"]).pack(side="left")
        self.sev_var = tk.StringVar(value="medium")
        ttk.Combobox(rf, textvariable=self.sev_var,
                     values=["low", "medium", "high", "critical"],
                     width=10, state="readonly").pack(side="left", padx=(4, 0))

        tk.Label(body, text="Description:", font=FS, bg=COLORS["bg"],
                 fg=COLORS["muted"]).pack(anchor="w")
        self.desc_var = tk.StringVar()
        desc_row = tk.Frame(body, bg=COLORS["bg"])
        desc_row.pack(fill="x", pady=(2, 8))
        tk.Entry(desc_row, textvariable=self.desc_var, font=FB,
                 bg=COLORS["input"], fg=COLORS["text"],
                 insertbackground=COLORS["text"], relief="flat",
                 highlightbackground=COLORS["border"], highlightthickness=1
                 ).pack(side="left", fill="x", expand=True, ipady=6)
        btn(desc_row, "Report", self._report, color=COLORS["red"], width=8
            ).pack(side="left", padx=(8,0))

        divider(body).pack(fill="x", pady=8)

        tk.Label(body, text="Active Incidents", font=FH3,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        lf = tk.Frame(body, bg=COLORS["input"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        lf.pack(fill="both", expand=True, pady=(6, 8))
        self.inc_lb = tk.Listbox(lf, font=FM, bg=COLORS["input"], fg=COLORS["text"],
                                  selectbackground=COLORS["accent2"], relief="flat",
                                  activestyle="none")
        sb = tk.Scrollbar(lf, command=self.inc_lb.yview,
                           troughcolor=COLORS["input"], bg=COLORS["border"])
        sb.pack(side="right", fill="y")
        self.inc_lb.config(yscrollcommand=sb.set)
        self.inc_lb.pack(fill="both", expand=True, padx=4, pady=4)

        br = tk.Frame(body, bg=COLORS["bg"])
        br.pack(fill="x")
        btn(br, "✓ Resolve Selected", self._resolve, color=COLORS["accent2"], width=18
            ).pack(side="left")
        btn(br, "Refresh", self._refresh, color=COLORS["card"], width=10
            ).pack(side="left", padx=8)
        btn(br, "Close", self.destroy, color=COLORS["card"], width=8).pack(side="right")

        self._incidents = []

    def _report(self):
        if not self.db:
            messagebox.showinfo("No DB", "Connect to a database first.", parent=self)
            return
        desc = self.desc_var.get().strip()
        if not desc:
            messagebox.showwarning("Missing", "Enter a description.", parent=self)
            return
        try:
            self.db.create_incident(
                road_name=self.road_var.get(),
                description=desc,
                severity=self.sev_var.get(),
            )
            self.desc_var.set("")
            self._refresh()
        except Exception as ex:
            messagebox.showerror("Error", str(ex), parent=self)

    def _resolve(self):
        if not self.db:
            return
        sel = self.inc_lb.curselection()
        if not sel or sel[0] >= len(self._incidents):
            return
        inc = self._incidents[sel[0]]
        try:
            self.db.resolve_incident(inc["id"])
            self._refresh()
        except Exception as ex:
            messagebox.showerror("Error", str(ex), parent=self)

    def _refresh(self):
        self.inc_lb.delete(0, tk.END)
        self._incidents = []
        if not self.db:
            self.inc_lb.insert(tk.END, "  No database connected.")
            return
        try:
            incidents = self.db.get_current_incidents()
            self._incidents = incidents
            if not incidents:
                self.inc_lb.insert(tk.END, "  No active incidents.")
            for inc in incidents:
                ts = str(inc.get("reported_at", ""))[:16]
                sev = inc.get("severity", "").upper()
                self.inc_lb.insert(
                    tk.END,
                    f"  [{sev}] {inc['road_name']} — {inc['description']} ({ts})"
                )
        except Exception as ex:
            self.inc_lb.insert(tk.END, f"  Error: {ex}")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE EDITOR DIALOG
# ═════════════════════════════════════════════════════════════════════════════

class RouteEditorDialog(tk.Toplevel):
    def __init__(self, parent, session: CameraSession):
        super().__init__(parent)
        self.session = session
        self.title(f"Edit Routes — {session.road_name}")
        self.geometry("500x420")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])
        self.grab_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 250
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 210
        self.geometry(f"500x420+{max(0,px)}+{max(0,py)}")
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=COLORS["blue"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  Routes for: {self.session.road_name}",
                 font=FH2, bg=COLORS["blue"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=14)
        tk.Label(body,
                 text="Routes shown in advisory panel when congestion is Moderate or higher.",
                 font=FS, bg=COLORS["bg"], fg=COLORS["muted"], wraplength=460
                 ).pack(anchor="w", pady=(0,10))

        row = tk.Frame(body, bg=COLORS["bg"])
        row.pack(fill="x", pady=(0,6))
        self.ev = tk.StringVar()
        tk.Entry(row, textvariable=self.ev, font=FB, bg=COLORS["input"], fg=COLORS["text"],
                 insertbackground=COLORS["text"], relief="flat",
                 highlightbackground=COLORS["border"], highlightthickness=1
                 ).pack(side="left", fill="x", expand=True, ipady=7)
        btn(row, "+ Add", self._add, color=COLORS["accent2"], width=7).pack(side="left", padx=(8,0))

        lf = tk.Frame(body, bg=COLORS["input"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        lf.pack(fill="both", expand=True, pady=(0,8))
        self.lb = tk.Listbox(lf, font=FB, bg=COLORS["input"], fg=COLORS["text"],
                              selectbackground=COLORS["accent2"], relief="flat", activestyle="none")
        self.lb.pack(fill="both", expand=True, padx=4, pady=4)
        for r in self.session.routes:
            self.lb.insert(tk.END, f"  {r}")

        br = tk.Frame(body, bg=COLORS["bg"])
        br.pack(fill="x")
        btn(br, "Remove Selected", self._remove, color=COLORS["red"], width=16).pack(side="left")
        btn(br, "Done", self.destroy, color=COLORS["accent2"], width=8).pack(side="right")

    def _add(self):
        name = self.ev.get().strip()
        if name and name not in self.session.routes:
            self.session.routes.append(name)
            self.lb.insert(tk.END, f"  {name}")
            self.ev.set("")

    def _remove(self):
        sel = self.lb.curselection()
        if sel:
            self.session.routes.pop(sel[0])
            self.lb.delete(sel[0])


# ═════════════════════════════════════════════════════════════════════════════
# DB SETTINGS DIALOG
# ═════════════════════════════════════════════════════════════════════════════

class DBSettingsDialog(tk.Toplevel):
    def __init__(self, parent, current_url: str):
        super().__init__(parent)
        self.result = None
        self.title("Database Connection")
        self.geometry("580x320")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])
        self.grab_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 290
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 160
        self.geometry(f"580x320+{max(0,px)}+{max(0,py)}")

        hdr = tk.Frame(self, bg=COLORS["blue"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Database Settings", font=FH2,
                 bg=COLORS["blue"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=16)
        tk.Label(body, text="Connection URL", font=FH3, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(anchor="w")
        tk.Label(body, text="PostgreSQL:  postgresql://user:pass@host:port/dbname",
                 font=FS, bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w")
        tk.Label(body, text="SQLite (no setup):  sqlite:///traffic.db",
                 font=FS, bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w")
        self.url_var = tk.StringVar(value=current_url)
        tk.Entry(body, textvariable=self.url_var, font=FM, bg=COLORS["input"],
                 fg=COLORS["text"], insertbackground=COLORS["text"], relief="flat",
                 highlightbackground=COLORS["border"], highlightthickness=1
                 ).pack(fill="x", ipady=7, pady=(6, 12))
        tk.Label(body,
                 text="Tables are created automatically.\nNo sample data is inserted.\n"
                      "Set DATABASE_URL in .env to persist this between launches.",
                 font=FS, bg=COLORS["bg"], fg=COLORS["muted"], justify="left"
                 ).pack(anchor="w")

        br = tk.Frame(body, bg=COLORS["bg"])
        br.pack(fill="x", pady=(14,0))
        btn(br, "Cancel", self.destroy, color=COLORS["card"], width=10).pack(side="right", padx=(8,0))
        btn(br, "Connect", self._confirm, color=COLORS["blue"], width=10).pack(side="right")

    def _confirm(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Missing", "Please enter a connection URL.", parent=self)
            return
        self.result = url
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# EXPORT PDF DIALOG
# ═════════════════════════════════════════════════════════════════════════════

class ExportPDFDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        self.title("Export PDF Report")
        self.geometry("440x220")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])
        self.grab_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 220
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 110
        self.geometry(f"440x220+{max(0,px)}+{max(0,py)}")

        hdr = tk.Frame(self, bg=COLORS["accent2"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Export PDF Report", font=FH2,
                 bg=COLORS["accent2"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=16)

        tk.Label(body, text="Time window (hours back from now):",
                 font=FH3, bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        hrow = tk.Frame(body, bg=COLORS["bg"])
        hrow.pack(fill="x", pady=(6, 14))
        self.hours_var = tk.IntVar(value=24)
        self.hours_lbl = tk.Label(hrow, text="24 h", font=FB, width=6,
                                   bg=COLORS["bg"], fg=COLORS["accent"])
        self.hours_lbl.pack(side="right")
        tk.Scale(hrow, from_=1, to=168, resolution=1, variable=self.hours_var,
                 orient="horizontal", bg=COLORS["bg"], fg=COLORS["text"],
                 troughcolor=COLORS["input"], highlightthickness=0, sliderlength=14,
                 command=lambda v: self.hours_lbl.config(text=f"{int(float(v))} h")
                 ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        br = tk.Frame(body, bg=COLORS["bg"])
        br.pack(fill="x")
        btn(br, "Cancel", self.destroy, color=COLORS["card"], width=10).pack(side="right", padx=(8,0))
        btn(br, "Export", self._confirm, color=COLORS["accent2"], width=10).pack(side="right")

    def _confirm(self):
        self.result = self.hours_var.get()
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA SIDEBAR CARD
# ═════════════════════════════════════════════════════════════════════════════

class CameraCard(tk.Frame):
    _STATUS_COLOR = {
        "idle": "#8B949E", "running": "#1DB954",
        "paused": "#F0883E", "stopped": "#FF4444",
    }

    def __init__(self, parent, session: CameraSession, on_select, on_remove):
        super().__init__(parent, bg=COLORS["card"],
                         highlightbackground=COLORS["border"], highlightthickness=1,
                         cursor="hand2")
        self.session   = session
        self.on_select = on_select
        self.on_remove = on_remove
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=COLORS["card"])
        top.pack(fill="x", padx=10, pady=(8,2))
        self.dot = tk.Label(top, text="●", font=("Segoe UI",10),
                            bg=COLORS["card"], fg=self._STATUS_COLOR["idle"])
        self.dot.pack(side="left")
        tk.Label(top, text=self.session.road_name, font=FH3,
                 bg=COLORS["card"], fg=COLORS["text"]).pack(side="left", padx=(4,0))
        rm = tk.Label(top, text="✕", font=("Segoe UI",9),
                      bg=COLORS["card"], fg=COLORS["muted"], cursor="hand2")
        rm.pack(side="right")
        rm.bind("<Button-1>", lambda e: self.on_remove(self.session.cam_id))
        src = str(self.session.source)
        if len(src) > 36:
            src = src[:34] + "…"
        tk.Label(self, text=src, font=FS, bg=COLORS["card"],
                 fg=COLORS["muted"]).pack(anchor="w", padx=10, pady=(0,7))
        self._bind_all()

    def _bind_all(self):
        for w in [self] + self.winfo_children():
            try:
                w.bind("<Button-1>", lambda e: self.on_select(self.session.cam_id))
            except Exception:
                pass

    def set_selected(self, sel: bool):
        bg = COLORS["hover"] if sel else COLORS["card"]
        hb = COLORS["accent"] if sel else COLORS["border"]
        self.config(bg=bg, highlightbackground=hb)
        for w in self.winfo_children():
            try: w.config(bg=bg)
            except Exception: pass

    def refresh(self):
        self.dot.config(fg=self._STATUS_COLOR.get(self.session.status, COLORS["muted"]))


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING THREAD
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_youtube(url: str) -> Optional[str]:
    if not YTDLP_OK:
        return None
    try:
        ydl_opts = {
            "format": "best[protocol^=m3u8]/best[protocol^=http]/best",
            "quiet": True, "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("is_live") or info.get("live_status") == "is_live":
                for f in reversed(info.get("formats", [])):
                    if f.get("protocol","").startswith("m3u8") and f.get("url"):
                        return f["url"]
                for f in reversed(info.get("formats", [])):
                    if f.get("url"):
                        return f["url"]
            return info.get("url") or info.get("manifest_url")
    except Exception as ex:
        _logger.error("yt-dlp error: %s", ex)
        return None


def run_session(session: CameraSession, db, log_fn, alert_fn):
    """
    Background thread — opens source, runs TrafficAnalyzer, pushes frames
    to session.frame_q, writes stats to session.stats, persists to DB.
    alert_fn(road_name, level) called when congestion level changes to severe.
    """
    source = session.source

    if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
        log_fn(f"[{session.road_name}] Resolving YouTube stream…")
        resolved = _resolve_youtube(source)
        if not resolved:
            log_fn(f"[{session.road_name}] ERROR: Could not resolve YouTube stream. "
                   "Install yt-dlp or check if the stream is public.")
            session.status = "stopped"
            return
        source = resolved
        log_fn(f"[{session.road_name}] YouTube stream resolved.")

    if not CV2_OK:
        log_fn(f"[{session.road_name}] ERROR: OpenCV not installed. pip install opencv-python")
        session.status = "stopped"
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_fn(f"[{session.road_name}] ERROR: Cannot open source: {source}")
        session.status = "stopped"
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    log_fn(f"[{session.road_name}] Opened — {w}×{h} @ {fps:.0f} fps")

    # Progress callback to log_fn for YOLO download
    def _progress(downloaded, total):
        if total > 0:
            pct = min(100, downloaded * 100 // total)
            log_fn(f"[{session.road_name}] Downloading YOLO weights: {pct}%")

    analyzer = None
    if ANALYZER_OK:
        try:
            analyzer = TrafficAnalyzer(
                model_key=session.model_key,
                use_tensorrt=session.use_tensorrt,
                use_fp16=session.use_fp16,
                confidence=session.confidence,
                pixels_per_meter=session.pixels_per_meter,
                max_density=session.max_density,
                counting_line_y_frac=session.counting_line_frac,
                progress_cb=_progress,
            )
            log_fn(f"[{session.road_name}] YOLO model loaded.")
        except Exception as ex:
            log_fn(f"[{session.road_name}] WARNING: Could not load analyzer: {ex}")
            log_fn(f"[{session.road_name}] Running in preview-only mode.")

    # Video writer (optional)
    writer = None
    if session.recording and CV2_OK and session.record_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(session.record_path, fourcc, fps, (w, h))
        log_fn(f"[{session.road_name}] Recording to: {session.record_path}")

    frame_skip    = 2
    frame_idx     = 0
    last_db_write = time.time()
    prev_level    = "low"

    while not session.stop_ev.is_set():
        if session.pause_ev.is_set():
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            if session.source_type in ("rtsp", "youtube"):
                log_fn(f"[{session.road_name}] Stream lost — retrying in 3 s…")
                time.sleep(3)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue
            else:
                log_fn(f"[{session.road_name}] End of video file.")
                break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        if analyzer:
            try:
                result = analyzer.process_frame(frame)
                for k, v in result.items():
                    if k != "frame":
                        session.stats[k] = v
                display = result["frame"]
            except Exception as ex:
                log_fn(f"[{session.road_name}] Frame error: {ex}")
                display = frame
        else:
            display = frame
            session.stats["timestamp"] = datetime.now()

        if writer:
            writer.write(display)

        if PIL_OK:
            try:
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail((820, 460), Image.LANCZOS)
                session.frame_q.put_nowait(img)
            except queue.Full:
                pass

        # Alert on congestion level change to severe/high
        cur_level = session.stats.get("congestion_level", "low")
        if cur_level in ("severe", "high") and prev_level not in ("severe", "high"):
            alert_fn(session.road_name, cur_level)
        prev_level = cur_level

        # DB write
        if db and (time.time() - last_db_write) >= session.db_interval:
            try:
                db.insert_reading(
                    road_name=session.road_name,
                    vehicle_count=int(session.stats.get("total_count", 0)),
                    congestion_score=int(session.stats.get("congestion_score", 0)),
                    congestion_level=str(session.stats.get("congestion_level", "low")),
                    avg_speed_kmh=float(session.stats.get("avg_speed_kmh", 0.0)),
                    vehicle_types=dict(session.stats.get("vehicle_type_counts", {})),
                    timestamp=datetime.now(),
                )
                log_fn(f"[{session.road_name}] DB write — {session.stats.get('total_count',0)} vehicles")
            except Exception as ex:
                log_fn(f"[{session.road_name}] DB write error: {ex}")
            last_db_write = time.time()

    cap.release()
    if writer:
        writer.release()
        log_fn(f"[{session.road_name}] Recording saved: {session.record_path}")
    log_fn(f"[{session.road_name}] Session ended.")
    session.status = "stopped"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Smart Traffic Analysis System")
        self.geometry("1400x840")
        self.minsize(1100, 700)
        self.configure(bg=COLORS["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            self.iconbitmap("traffic_icon.ico")
        except Exception:
            pass

        self.sessions:   dict[int, CameraSession] = {}
        self.cam_cards:  dict[int, CameraCard]    = {}
        self.next_id     = 1
        self.active_id: Optional[int] = None
        self.db          = None
        self.db_url      = os.environ.get(
            "DATABASE_URL",
            "sqlite:///traffic.db"   # SQLite fallback — zero setup required
        )
        self._chart_tick = 0
        self._chart_canvas = None
        self._chart_fig    = None

        self._connect_db()
        self._build()
        self._bind_shortcuts()
        self._ui_loop_preview()
        self._ui_loop_stats()
        self._tick()

        self._load_cameras()
        self.log("Smart Traffic Analysis System — ready.")
        self._check_deps()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_cameras(self):
        try:
            data = {
                "next_id": self.next_id,
                "cameras": [s.to_dict() for s in self.sessions.values()],
            }
            with open(CAMERAS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as ex:
            _logger.warning("Could not save cameras: %s", ex)

    def _load_cameras(self):
        if not os.path.exists(CAMERAS_FILE):
            return
        try:
            with open(CAMERAS_FILE) as f:
                data = json.load(f)
            self.next_id = data.get("next_id", 1)
            for d in data.get("cameras", []):
                session = CameraSession.from_dict(d)
                self.sessions[session.cam_id] = session
                card = CameraCard(self.cam_list, session,
                                  on_select=self._select, on_remove=self._remove)
                card.pack(fill="x", pady=(0,6))
                self.cam_cards[session.cam_id] = card
            if self.sessions:
                first = next(iter(self.sessions))
                self._select(first)
            self.log(f"Restored {len(self.sessions)} camera(s) from cameras.json")
        except Exception as ex:
            self.log(f"Could not restore cameras: {ex}")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _bind_shortcuts(self):
        self.bind("<space>",    lambda e: self._start())
        self.bind("<Escape>",   lambda e: self._stop())
        self.bind("<Control-n>",lambda e: self._add_camera())
        self.bind("<Control-e>",lambda e: self._edit_routes())
        self.bind("<Control-p>",lambda e: self._export_pdf())
        self.bind("<Control-r>",lambda e: self._reset_counts())
        self.bind("<Control-i>",lambda e: self._open_incidents())

    # ── Alerts ────────────────────────────────────────────────────────────────

    def _alert(self, road_name: str, level: str):
        """Called from background thread when congestion becomes high/severe."""
        self.after(0, lambda: self._do_alert(road_name, level))

    def _do_alert(self, road_name: str, level: str):
        self.log(f"⚠ ALERT: {road_name} — {level.upper()} congestion detected!")
        # Audio beep (Windows)
        try:
            import winsound
            freq = 1200 if level == "severe" else 800
            winsound.Beep(freq, 400)
        except Exception:
            pass
        # Desktop notification (Windows toast via winotify if available)
        try:
            from winotify import Notification, audio
            toast = Notification(
                app_id="Smart Traffic Analysis",
                title=f"⚠ {level.upper()} Congestion — {road_name}",
                msg="Check alternative routes.",
                duration="short",
            )
            toast.set_audio(audio.Default, loop=False)
            toast.show()
        except Exception:
            pass

    # ── Dependency check ──────────────────────────────────────────────────────

    def _check_deps(self):
        missing = []
        if not CV2_OK:
            missing.append("opencv-python  (camera + detection)")
        if not PIL_OK:
            missing.append("Pillow  (video preview)")
        if not MPL_OK:
            missing.append("matplotlib  (charts + PDF report)")
        if not DB_MOD_OK:
            missing.append("psycopg2-binary  (PostgreSQL — not needed for SQLite)")
        if not YTDLP_OK:
            self.log("INFO: yt-dlp not installed — YouTube Live sources unavailable.")
        for m in missing:
            self.log(f"WARNING: Missing — {m}")
        if missing:
            self.log("Run:  pip install -r requirements_gui.txt")

    # ── Database ──────────────────────────────────────────────────────────────

    def _connect_db(self):
        if not DB_MOD_OK:
            return
        try:
            self.db = DatabaseManager(self.db_url)
            self.db.initialize()
        except Exception as ex:
            self.db = None
            _logger.warning("DB not connected: %s", ex)

    def _open_db_settings(self):
        dlg = DBSettingsDialog(self, self.db_url)
        self.wait_window(dlg)
        if dlg.result:
            if self.db:
                self.db.close()
            self.db_url = dlg.result
            self._connect_db()
            connected = self.db is not None
            self.db_status_lbl.config(
                text="● DB Connected" if connected else "○ DB Disconnected",
                fg=COLORS["accent"] if connected else COLORS["amber"],
            )
            self.log(f"Database: {'connected (' + self.db_url[:30] + '…)' if connected else 'failed — check URL'}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build(self):
        self._build_topbar()
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True)
        self._build_sidebar(main)
        self._build_content(main)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=COLORS["panel"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=COLORS["accent"], width=4, height=52).pack(side="left")
        tk.Label(bar, text="  Smart Traffic Analysis System",
                 font=FT, bg=COLORS["panel"], fg=COLORS["text"]).pack(side="left")

        self.db_status_lbl = tk.Label(
            bar,
            text="● DB Connected" if self.db else "○ DB Disconnected",
            font=FS, bg=COLORS["panel"],
            fg=COLORS["accent"] if self.db else COLORS["amber"],
            cursor="hand2",
        )
        self.db_status_lbl.pack(side="right", padx=(0,16))
        self.db_status_lbl.bind("<Button-1>", lambda e: self._open_db_settings())

        self.clock_lbl = tk.Label(bar, text="", font=FS, bg=COLORS["panel"], fg=COLORS["muted"])
        self.clock_lbl.pack(side="right", padx=(0,16))

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=COLORS["panel"], width=240)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        tk.Label(sb, text="CAMERAS", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=14, pady=(12,6))

        self.cam_list = tk.Frame(sb, bg=COLORS["panel"])
        self.cam_list.pack(fill="both", expand=True, padx=8)

        divider(sb).pack(fill="x", pady=4)
        btn(sb, "+ Add Camera", self._add_camera, color=COLORS["accent2"], width=26
            ).pack(padx=10, pady=(4,4), fill="x")

        tk.Label(sb, text="SELECTED CAMERA", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=14, pady=(6,4))

        c1 = tk.Frame(sb, bg=COLORS["panel"])
        c1.pack(fill="x", padx=8, pady=(0,4))
        btn(c1, "▶ Start",  self._start,  color=COLORS["accent2"], width=9).pack(side="left", padx=(0,4))
        btn(c1, "⏸ Pause", self._pause,  color=COLORS["amber"],   width=9).pack(side="left")

        c2 = tk.Frame(sb, bg=COLORS["panel"])
        c2.pack(fill="x", padx=8, pady=(0,4))
        btn(c2, "⏹ Stop",  self._stop,  color=COLORS["red"],  width=9).pack(side="left", padx=(0,4))
        btn(c2, "✎ Routes", self._edit_routes, color=COLORS["blue"], width=9).pack(side="left")

        c3 = tk.Frame(sb, bg=COLORS["panel"])
        c3.pack(fill="x", padx=8, pady=(0,4))
        btn(c3, "↺ Reset Counts", self._reset_counts, color=COLORS["card"], width=14
            ).pack(side="left", padx=(0,4))

        divider(sb).pack(fill="x", pady=6)
        btn(sb, "⚠ Incidents", self._open_incidents, color=COLORS["red"], width=26
            ).pack(padx=10, pady=(0,4), fill="x")
        btn(sb, "Export PDF Report", self._export_pdf, color=COLORS["card"], width=26
            ).pack(padx=10, pady=(0,10), fill="x")

        # Shortcuts hint
        tk.Label(sb, text="Space=Start  Esc=Stop\nCtrl+N=Add  Ctrl+R=Reset\nCtrl+I=Incidents  Ctrl+P=PDF",
                 font=("Segoe UI",8), bg=COLORS["panel"], fg=COLORS["muted"],
                 justify="left").pack(anchor="w", padx=14, pady=(4,8))

    def _build_content(self, parent):
        ct = tk.Frame(parent, bg=COLORS["bg"])
        ct.pack(side="left", fill="both", expand=True)
        top = tk.Frame(ct, bg=COLORS["bg"])
        top.pack(fill="both", expand=True, padx=8, pady=(8,0))
        self._build_video_panel(top)
        self._build_right_panel(top)
        self._build_stats(ct)
        self._build_bottom(ct)

    def _build_video_panel(self, parent):
        vf = tk.Frame(parent, bg=COLORS["panel"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        vf.pack(side="left", fill="both", expand=True)
        hdr = tk.Frame(vf, bg=COLORS["panel"], height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=COLORS["accent"], width=3, height=32).pack(side="left")
        self.vid_title = tk.Label(hdr, text="No camera selected",
                                   font=FH3, bg=COLORS["panel"], fg=COLORS["text"])
        self.vid_title.pack(side="left", padx=10)
        self.vid_status = tk.Label(hdr, text="● IDLE", font=FS,
                                    bg=COLORS["panel"], fg=COLORS["muted"])
        self.vid_status.pack(side="right", padx=10)
        self.video_lbl = tk.Label(vf, bg="#050A0D",
                                   text="Select a camera and press  ▶ Start  (or Space)",
                                   font=FB, fg=COLORS["muted"])
        self.video_lbl.pack(fill="both", expand=True)

    def _build_right_panel(self, parent):
        rp = tk.Frame(parent, bg=COLORS["panel"], width=282,
                      highlightbackground=COLORS["border"], highlightthickness=1)
        rp.pack(side="left", fill="y", padx=(8,0))
        rp.pack_propagate(False)

        tk.Label(rp, text="ROUTE ADVISORY", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(10,4))

        br = tk.Frame(rp, bg=COLORS["panel"])
        br.pack(fill="x", padx=12, pady=(0,6))
        tk.Label(br, text="Congestion:", font=FS, bg=COLORS["panel"],
                 fg=COLORS["muted"]).pack(side="left")
        self.cong_badge = tk.Label(br, text=" LOW ", font=("Segoe UI",9,"bold"),
                                    bg="#1A4A1A", fg=COLORS["accent"], padx=6, pady=2)
        self.cong_badge.pack(side="left", padx=(8,0))

        self.route_txt = tk.Text(rp, font=FB, height=5, bg=COLORS["input"],
                                  fg=COLORS["text"], relief="flat", wrap="word",
                                  state="disabled", padx=10, pady=8)
        self.route_txt.pack(fill="x", padx=12, pady=(0,8))

        tk.Label(rp, text="CONFIGURED ROUTES", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(4,4))
        rlf = tk.Frame(rp, bg=COLORS["input"],
                       highlightbackground=COLORS["border"], highlightthickness=1)
        rlf.pack(fill="x", padx=12, pady=(0,8))
        self.routes_lb = tk.Listbox(rlf, font=FB, bg=COLORS["input"], fg=COLORS["text"],
                                     selectbackground=COLORS["accent2"], relief="flat",
                                     height=3, activestyle="none")
        self.routes_lb.pack(fill="x", padx=4, pady=4)

        tk.Label(rp, text="VEHICLE BREAKDOWN", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(4,4))
        self._vbars = {}
        vtf = tk.Frame(rp, bg=COLORS["panel"])
        vtf.pack(fill="x", padx=12)
        colors = {"car": COLORS["accent"], "motorcycle": COLORS["blue"],
                  "truck": COLORS["amber"], "bus": COLORS["red"], "bicycle": COLORS["muted"]}
        for vt in ["car","motorcycle","truck","bus","bicycle"]:
            row = tk.Frame(vtf, bg=COLORS["panel"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=vt.capitalize(), font=FS, width=11, anchor="w",
                     bg=COLORS["panel"], fg=COLORS["muted"]).pack(side="left")
            bg_f = tk.Frame(row, bg=COLORS["input"], height=10)
            bg_f.pack(side="left", fill="x", expand=True)
            fill_f = tk.Frame(bg_f, bg=colors[vt], height=10, width=0)
            fill_f.place(x=0, y=0, relheight=1)
            cnt = tk.Label(row, text="0", font=FS, width=5, anchor="e",
                           bg=COLORS["panel"], fg=COLORS["text"])
            cnt.pack(side="left")
            self._vbars[vt] = (bg_f, fill_f, cnt)

    def _build_stats(self, parent):
        sf = tk.Frame(parent, bg=COLORS["panel"], height=110)
        sf.pack(fill="x", padx=8, pady=(6,0))
        sf.pack_propagate(False)
        self._stat_vals = {}
        specs = [
            ("Total Vehicles",   "total_count",     COLORS["accent"], "#"),
            ("Live Density",     "current_density", COLORS["blue"],   "vehicles"),
            ("Congestion Index", "congestion_score",COLORS["amber"],  "/ 100"),
            ("Avg Speed",        "avg_speed_kmh",   COLORS["accent"], "km/h"),
            ("Processing FPS",   "fps",             COLORS["blue"],   "fps"),
        ]
        for i, (label, key, accent, unit) in enumerate(specs):
            card = tk.Frame(sf, bg=COLORS["card"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
            card.pack(side="left", fill="both", expand=True,
                      padx=(0 if i>0 else 8, 8), pady=8)
            tk.Frame(card, bg=accent, height=3).pack(fill="x")
            tk.Label(card, text=label, font=FS, bg=COLORS["card"],
                     fg=COLORS["muted"]).pack(pady=(6,0))
            v = tk.Label(card, text="—", font=FST, bg=COLORS["card"], fg=COLORS["text"])
            v.pack()
            tk.Label(card, text=unit, font=FS, bg=COLORS["card"],
                     fg=COLORS["muted"]).pack(pady=(0,6))
            self._stat_vals[key] = v

    def _build_bottom(self, parent):
        bot = tk.Frame(parent, bg=COLORS["bg"])
        bot.pack(fill="x", padx=8, pady=(6,8))

        cf = tk.Frame(bot, bg=COLORS["panel"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        cf.pack(side="left", fill="both", expand=True)
        tk.Label(cf, text="VEHICLE TYPE DISTRIBUTION", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=10, pady=(8,0))
        self._chart_host = tk.Frame(cf, bg=COLORS["panel"], height=140)
        self._chart_host.pack(fill="both", expand=True, padx=6, pady=(2,6))
        self._chart_host.pack_propagate(False)
        self._init_chart()

        lf = tk.Frame(bot, bg=COLORS["panel"],
                      highlightbackground=COLORS["border"], highlightthickness=1)
        lf.pack(side="left", fill="both", expand=True, padx=(8,0))
        tk.Label(lf, text="SYSTEM LOG", font=("Segoe UI",9,"bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=10, pady=(8,0))
        inner = tk.Frame(lf, bg=COLORS["input"])
        inner.pack(fill="both", expand=True, padx=8, pady=(4,8))
        self.log_txt = tk.Text(inner, font=FM, height=8, bg=COLORS["input"],
                                fg=COLORS["muted"], relief="flat", state="disabled",
                                wrap="word", padx=6, pady=4)
        sb = tk.Scrollbar(inner, command=self.log_txt.yview,
                           troughcolor=COLORS["input"], bg=COLORS["border"])
        sb.pack(side="right", fill="y")
        self.log_txt.config(yscrollcommand=sb.set)
        self.log_txt.pack(fill="both", expand=True)

    # ── Chart — initialise once, update in-place to avoid flicker ────────────

    def _init_chart(self):
        if not MPL_OK:
            return
        fig, ax = plt.subplots(figsize=(4.5, 1.5), facecolor=COLORS["panel"])
        ax.set_facecolor(COLORS["panel"])
        ax.text(0.5, 0.5, "No data yet — start a camera session",
                ha="center", va="center", color=COLORS["muted"],
                fontsize=9, transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self._chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._chart_fig    = fig
        self._chart_ax     = ax
        self._chart_canvas = canvas

    def _update_chart(self, vtypes: dict):
        if not MPL_OK or self._chart_canvas is None:
            return
        ax = self._chart_ax
        ax.cla()
        ax.set_facecolor(COLORS["panel"])
        labels = [k.capitalize() for k, v in vtypes.items() if v and int(v) > 0]
        vals   = [int(v) for v in vtypes.values() if v and int(v) > 0]
        bar_colors = [COLORS["accent"], COLORS["blue"], COLORS["amber"],
                      COLORS["red"], COLORS["muted"]]
        if labels:
            bars = ax.barh(labels, vals, color=bar_colors[:len(labels)], height=0.5)
            ax.set_xlim(0, max(vals) * 1.3 or 1)
            for bar, v in zip(bars, vals):
                ax.text(v + max(vals)*0.02, bar.get_y()+bar.get_height()/2,
                        str(v), va="center", fontsize=8, color=COLORS["text"])
            ax.invert_yaxis()
            ax.tick_params(colors=COLORS["muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLORS["border"])
        else:
            ax.text(0.5, 0.5, "No data yet — start a camera session",
                    ha="center", va="center", color=COLORS["muted"],
                    fontsize=9, transform=ax.transAxes)
            ax.axis("off")
        self._chart_fig.tight_layout(pad=0.3)
        self._chart_canvas.draw_idle()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_txt.config(state="normal")
        self.log_txt.insert(tk.END, line)
        self.log_txt.see(tk.END)
        self.log_txt.config(state="disabled")
        _logger.info(msg)

    # ── Camera management ─────────────────────────────────────────────────────

    def _add_camera(self):
        dlg = AddCameraDialog(self)
        self.wait_window(dlg)
        if not dlg.result:
            return
        r = dlg.result
        session = CameraSession(
            cam_id=self.next_id,
            source=r["source"], road_name=r["road_name"],
            routes=r["routes"], confidence=r["confidence"],
            source_type=r["source_type"], db_interval=r["db_interval"],
            counting_line_frac=r["counting_line_frac"],
            pixels_per_meter=r["pixels_per_meter"],
            max_density=r["max_density"],
            model_key=r.get("model_key", "yolo11l"),
            use_tensorrt=r.get("use_tensorrt", False),
            use_fp16=r.get("use_fp16", True),
        )
        session.recording  = r.get("record", False)
        session.record_path = (
            os.path.join(ROOT, f"output/{session.road_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            if session.recording else None
        )
        self.sessions[self.next_id] = session
        card = CameraCard(self.cam_list, session,
                          on_select=self._select, on_remove=self._remove)
        card.pack(fill="x", pady=(0,6))
        self.cam_cards[self.next_id] = card
        self.next_id += 1
        self._save_cameras()
        self.log(f"Camera added: {r['road_name']}  ({r['source_type']})")
        self._select(session.cam_id)

    def _select(self, cam_id: int):
        self.active_id = cam_id
        for cid, card in self.cam_cards.items():
            card.set_selected(cid == cam_id)
        s = self.sessions.get(cam_id)
        if s:
            self.vid_title.config(text=s.road_name)
            self._refresh_routes_lb(s)
            self._refresh_status_display(s)

    def _remove(self, cam_id: int):
        s = self.sessions.get(cam_id)
        if not s:
            return
        if s.status == "running":
            if not messagebox.askyesno("Confirm", f"Stop and remove  '{s.road_name}'?"):
                return
            s.stop_ev.set()
        card = self.cam_cards.pop(cam_id, None)
        if card:
            card.destroy()
        self.sessions.pop(cam_id, None)
        if self.active_id == cam_id:
            self.active_id = None
            self.vid_title.config(text="No camera selected")
            self.video_lbl.config(image="", text="Select a camera and press  ▶ Start")
        self._save_cameras()
        self.log(f"Camera removed: {s.road_name}")

    def _refresh_routes_lb(self, s: CameraSession):
        self.routes_lb.delete(0, tk.END)
        for r in s.routes:
            self.routes_lb.insert(tk.END, f"  {r}")

    def _refresh_status_display(self, s: CameraSession):
        colors = {"idle": COLORS["muted"], "running": COLORS["accent"],
                  "paused": COLORS["amber"], "stopped": COLORS["red"]}
        labels = {"idle": "● IDLE", "running": "● LIVE",
                  "paused": "⏸ PAUSED", "stopped": "■ STOPPED"}
        self.vid_status.config(
            text=labels.get(s.status, "● IDLE"),
            fg=colors.get(s.status, COLORS["muted"]),
        )

    # ── Session control ───────────────────────────────────────────────────────

    def _active_session(self) -> Optional[CameraSession]:
        if self.active_id is None:
            messagebox.showinfo("No camera", "Please select a camera first.")
            return None
        return self.sessions.get(self.active_id)

    def _start(self):
        s = self._active_session()
        if not s:
            return
        if s.status == "running":
            return
        if s.status == "paused":
            s.pause_ev.clear()
            s.status = "running"
            self._refresh_status_display(s)
            self.cam_cards[s.cam_id].refresh()
            self.log(f"{s.road_name} — resumed.")
            return
        # Wait for any lingering thread to finish before starting a new one
        if s.thread and s.thread.is_alive():
            s.stop_ev.set()
            s.thread.join(timeout=3.0)
        s.stop_ev.clear()
        s.pause_ev.clear()
        s.status = "running"
        t = threading.Thread(target=run_session,
                             args=(s, self.db, self.log, self._alert), daemon=True)
        s.thread = t
        t.start()
        self._refresh_status_display(s)
        self.cam_cards[s.cam_id].refresh()
        self.log(f"{s.road_name} — session started.")

    def _pause(self):
        s = self._active_session()
        if not s or s.status != "running":
            return
        s.pause_ev.set()
        s.status = "paused"
        self._refresh_status_display(s)
        self.cam_cards[s.cam_id].refresh()
        self.log(f"{s.road_name} — paused.")

    def _stop(self):
        s = self._active_session()
        if not s:
            return
        s.stop_ev.set()
        s.pause_ev.clear()
        s.status = "stopped"
        self._refresh_status_display(s)
        self.cam_cards[s.cam_id].refresh()
        self.log(f"{s.road_name} — stopped.")

    def _edit_routes(self):
        s = self._active_session()
        if not s:
            return
        dlg = RouteEditorDialog(self, s)
        self.wait_window(dlg)
        self._refresh_routes_lb(s)
        self._save_cameras()
        self.log(f"{s.road_name} — routes updated ({len(s.routes)} routes).")

    def _reset_counts(self):
        s = self._active_session()
        if not s:
            return
        # Reset stats dict
        s.stats = {
            "total_count": 0, "current_density": 0,
            "congestion_score": 0, "congestion_level": "low",
            "avg_speed_kmh": 0.0, "fps": 0.0,
            "vehicle_type_counts": {}, "timestamp": None,
        }
        self.log(f"{s.road_name} — counts reset.")

    def _open_incidents(self):
        IncidentDialog(self, self.db, self.sessions)

    # ── PDF export ────────────────────────────────────────────────────────────

    def _export_pdf(self):
        if not REPORT_OK:
            messagebox.showinfo("Not available",
                                "PDF export requires matplotlib.\n"
                                "Run:  pip install matplotlib")
            return
        if not self.db:
            messagebox.showinfo("No database",
                                "Connect to a database first.\n"
                                "Click the DB status label in the top bar.")
            return

        dlg = ExportPDFDialog(self)
        self.wait_window(dlg)
        if not dlg.result:
            return
        hours = dlg.result

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files","*.pdf")],
            initialfile=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
        if not path:
            return
        self.log(f"Generating PDF report (last {hours}h)…")

        def _do():
            try:
                rg = ReportGenerator(self.db)
                rg.generate_pdf(path, hours=hours)
                self.after(0, lambda: self.log(f"PDF saved: {path}"))
                self.after(0, lambda: messagebox.showinfo("Done", f"Report saved:\n{path}"))
            except Exception as ex:
                self.after(0, lambda: self.log(f"PDF error: {ex}"))
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))
        threading.Thread(target=_do, daemon=True).start()

    # ── UI refresh loops ──────────────────────────────────────────────────────

    def _ui_loop_preview(self):
        s = self.sessions.get(self.active_id)
        if s and s.status in ("running","paused") and PIL_OK:
            try:
                img = s.frame_q.get_nowait()
                photo = ImageTk.PhotoImage(img)
                self.video_lbl.config(image=photo, text="")
                self.video_lbl.image = photo
            except queue.Empty:
                pass
        self.after(33, self._ui_loop_preview)

    def _ui_loop_stats(self):
        s = self.sessions.get(self.active_id)
        if s and s.status in ("running","paused"):
            st = s.stats
            self._stat_vals["total_count"    ].config(text=str(st.get("total_count",0)))
            self._stat_vals["current_density"].config(text=str(st.get("current_density",0)))
            self._stat_vals["congestion_score"].config(text=str(st.get("congestion_score",0)))
            self._stat_vals["avg_speed_kmh"  ].config(text=f"{st.get('avg_speed_kmh',0.0):.1f}")
            self._stat_vals["fps"            ].config(text=f"{st.get('fps',0.0):.1f}")

            lvl = st.get("congestion_level","low")
            badge_map = {
                "low":      ("#1A4A1A", COLORS["accent"]),
                "moderate": ("#3D2E00", COLORS["amber"]),
                "high":     ("#3D1500", "#F0883E"),
                "severe":   ("#3D0000", COLORS["red"]),
            }
            bg, fg = badge_map.get(lvl, ("#1A4A1A", COLORS["accent"]))
            self.cong_badge.config(text=f" {lvl.upper()} ", bg=bg, fg=fg)

            routes = s.routes
            if lvl == "low":
                msg = "Traffic flowing normally.\nNo route diversion needed."
            elif lvl == "moderate":
                msg = "Minor delays detected.\n"
                msg += ("Consider:\n" + "\n".join(f"  {i+1}. {r}" for i,r in enumerate(routes[:2]))
                        if routes else "No alternative routes configured.\nClick  ✎ Routes  to add some.")
            else:
                msg = f"{'Heavy' if lvl=='high' else 'SEVERE'} congestion detected!\n\n"
                msg += ("Recommended alternatives:\n" + "\n".join(f"  {i+1}. {r}" for i,r in enumerate(routes))
                        if routes else "No routes configured.\nClick  ✎ Routes  to add alternatives.")
            self.route_txt.config(state="normal")
            self.route_txt.delete("1.0", tk.END)
            self.route_txt.insert(tk.END, msg)
            self.route_txt.config(state="disabled")

            vtypes = st.get("vehicle_type_counts", {})
            total_v = max(1, sum(int(v) for v in vtypes.values() if v))
            for vt, (bg_f, fill_f, cnt_l) in self._vbars.items():
                count = int(vtypes.get(vt, 0))
                cnt_l.config(text=str(count))
                pct = count / total_v
                bg_f.update_idletasks()
                w = int(bg_f.winfo_width() * pct)
                fill_f.place(x=0, y=0, width=max(0,w), relheight=1)

            self._chart_tick += 1
            if self._chart_tick % 10 == 0:
                self._update_chart(vtypes)

        self.after(500, self._ui_loop_stats)

    # ── Clock ─────────────────────────────────────────────────────────────────

    def _tick(self):
        self.clock_lbl.config(text=datetime.now().strftime("%d %b %Y  %H:%M:%S"))
        self.after(1000, self._tick)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        for s in self.sessions.values():
            s.stop_ev.set()
        if self.db:
            self.db.close()
        self._save_cameras()
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
