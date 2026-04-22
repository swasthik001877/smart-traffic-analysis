"""
Smart Traffic Analysis System — Complete Integrated GUI
=======================================================
Single-file Windows desktop application.

Integrates with:
  • core/traffic_analyzer.py   — YOLO vehicle detection + tracking
  • database/db_manager.py     — PostgreSQL persistence
  • analysis/report_generator.py — PDF report export

All modules are imported using sys.path injection so this file
can sit in the project root alongside core/, database/, etc.

Run:  python gui_app.py
"""

import sys
import os

# ── Ensure the project root is on sys.path so all sub-packages resolve ───────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import json
from datetime import datetime
from typing import Optional

# ── Optional heavy imports with graceful fallback ────────────────────────────
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

# ── Project module imports ────────────────────────────────────────────────────
try:
    from database.db_manager import DatabaseManager
    DB_MOD_OK = True
except Exception as _e:
    DB_MOD_OK = False
    DatabaseManager = None

try:
    from core.traffic_analyzer import TrafficAnalyzer
    ANALYZER_OK = True
except Exception as _e:
    ANALYZER_OK = False
    TrafficAnalyzer = None

try:
    from analysis.report_generator import ReportGenerator
    REPORT_OK = True
except Exception as _e:
    REPORT_OK = False
    ReportGenerator = None

try:
    from utils.logger import setup_logger
    _logger = setup_logger("gui")
except Exception:
    import logging
    _logger = logging.getLogger("gui")


# ═════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═════════════════════════════════════════════════════════════════════════════

P = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "card":     "#1C2128",
    "input":    "#21262D",
    "accent":   "#1DB954",
    "accent2":  "#238636",
    "blue":     "#388BFD",
    "amber":    "#F0883E",
    "red":      "#FF4444",
    "text":     "#E6EDF3",
    "muted":    "#8B949E",
    "border":   "#30363D",
    "hover":    "#2D333B",
}

FT = ("Segoe UI", 22, "bold")
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
    c = color or P["accent2"]
    b = tk.Button(parent, text=text, command=cmd,
                  bg=c, fg=P["text"], font=FB,
                  relief="flat", bd=0, padx=12, pady=6,
                  width=width, cursor="hand2",
                  activebackground=P["accent"],
                  activeforeground="#fff", **kw)
    b.bind("<Enter>", lambda e: b.config(bg=P["accent"]))
    b.bind("<Leave>", lambda e: b.config(bg=c))
    return b


def lbl(parent, text, font=FB, fg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or P["text"], bg=P["panel"], **kw)


def divider(parent, color=None):
    return tk.Frame(parent, bg=color or P["border"], height=1)


# ═════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

class CameraSession:
    def __init__(self, cam_id, source, road_name, routes, confidence=0.5,
                 source_type="webcam"):
        self.cam_id      = cam_id
        self.source      = source
        self.road_name   = road_name
        self.routes      = list(routes)
        self.confidence  = confidence
        self.source_type = source_type
        self.status      = "idle"
        self.thread      = None
        self.stop_ev     = threading.Event()
        self.pause_ev    = threading.Event()
        self.frame_q     = queue.Queue(maxsize=2)
        self.stats = {
            "total_count": 0, "current_density": 0,
            "congestion_score": 0, "congestion_level": "low",
            "avg_speed_kmh": 0.0, "fps": 0.0,
            "vehicle_type_counts": {}, "timestamp": None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# ADD CAMERA DIALOG
# ═════════════════════════════════════════════════════════════════════════════

class AddCameraDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        self.routes = []
        self.title("Add Camera / Video Source")
        self.geometry("600x660")
        self.resizable(False, False)
        self.configure(bg=P["bg"])
        self.grab_set()
        self.focus_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 300
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 330
        self.geometry(f"600x660+{max(0,px)}+{max(0,py)}")
        self._build()

    def _build(self):
        # Header strip
        hdr = tk.Frame(self, bg=P["accent2"], height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  Add Camera / Video Source",
                 font=FH2, bg=P["accent2"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=14)

        # Source type radio
        self._section(body, "Source Type")
        self.src_type = tk.StringVar(value="webcam")
        types_row = tk.Frame(body, bg=P["bg"])
        types_row.pack(fill="x", pady=(4, 12))
        for val, label in [("webcam", "Webcam"), ("rtsp", "RTSP / IP Camera"),
                            ("youtube", "YouTube Live"), ("file", "Video File")]:
            tk.Radiobutton(types_row, text=label, variable=self.src_type,
                           value=val, command=self._on_type,
                           bg=P["bg"], fg=P["text"], selectcolor=P["input"],
                           activebackground=P["bg"], font=FB
                           ).pack(side="left", padx=(0, 14))

        # Dynamic source input area
        self.src_area = tk.Frame(body, bg=P["bg"])
        self.src_area.pack(fill="x", pady=(0, 12))
        self._build_webcam()

        # Road name
        self._section(body, "Road / Location Name")
        self.road_var = tk.StringVar()
        self._entry(body, self.road_var).pack(fill="x", ipady=6, pady=(3, 12))

        # Routes
        self._section(body, "Alternative Routes  (suggested when congestion is high)")
        row = tk.Frame(body, bg=P["bg"])
        row.pack(fill="x", pady=(3, 5))
        self.route_var = tk.StringVar()
        self._entry(row, self.route_var).pack(side="left", fill="x", expand=True, ipady=6)
        btn(row, "+ Add", self._add_route, color=P["blue"], width=7
            ).pack(side="left", padx=(8, 0))

        # Hint
        tk.Label(body,
                 text='Example: "Ring Road North → City Bypass (+4 min)"',
                 font=FS, bg=P["bg"], fg=P["muted"]
                 ).pack(anchor="w", pady=(0, 4))

        # Route listbox
        lf = tk.Frame(body, bg=P["input"],
                      highlightbackground=P["border"], highlightthickness=1)
        lf.pack(fill="x", pady=(0, 4))
        self.route_lb = tk.Listbox(lf, font=FB, bg=P["input"], fg=P["text"],
                                    selectbackground=P["accent2"], relief="flat",
                                    height=4, activestyle="none")
        self.route_lb.pack(fill="x", padx=4, pady=4)
        btn(body, "Remove Selected", self._remove_route,
            color=P["card"], width=16).pack(anchor="w", pady=(0, 10))

        # Confidence slider
        cf = tk.Frame(body, bg=P["bg"])
        cf.pack(fill="x", pady=(0, 14))
        tk.Label(cf, text="Detection Confidence:", font=FH3,
                 bg=P["bg"], fg=P["text"]).pack(side="left")
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_lbl = tk.Label(cf, text="0.50", font=FB, width=4,
                                  bg=P["bg"], fg=P["accent"])
        self.conf_lbl.pack(side="right")
        tk.Scale(cf, from_=0.1, to=0.9, resolution=0.05,
                 variable=self.conf_var, orient="horizontal",
                 bg=P["bg"], fg=P["text"], troughcolor=P["input"],
                 highlightthickness=0, sliderlength=14, length=180,
                 command=lambda v: self.conf_lbl.config(text=f"{float(v):.2f}")
                 ).pack(side="right", padx=8)

        # Buttons
        br = tk.Frame(body, bg=P["bg"])
        br.pack(fill="x")
        btn(br, "Cancel", self.destroy, color=P["card"], width=10).pack(side="right", padx=(8, 0))
        btn(br, "Add Camera", self._confirm, color=P["accent2"], width=12).pack(side="right")

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=FH3, bg=P["bg"], fg=P["text"]
                 ).pack(anchor="w")

    def _entry(self, parent, var):
        return tk.Entry(parent, textvariable=var, font=FB,
                        bg=P["input"], fg=P["text"],
                        insertbackground=P["text"], relief="flat",
                        highlightbackground=P["border"], highlightthickness=1)

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
                 font=FS, bg=P["bg"], fg=P["muted"]).pack(anchor="w")
        self.src_var = tk.StringVar(value="0")
        self._entry(self.src_area, self.src_var).pack(anchor="w", ipady=6, pady=(3, 0), ipadx=20)

    def _build_url(self, label, placeholder, warn=False, warn_msg=""):
        self._clear_src()
        tk.Label(self.src_area, text=label, font=FH3, bg=P["bg"], fg=P["text"]
                 ).pack(anchor="w")
        if warn:
            tk.Label(self.src_area, text=warn_msg, font=FS, bg=P["bg"],
                     fg=P["amber"]).pack(anchor="w")
        self.src_var = tk.StringVar(value="")
        self._entry(self.src_area, self.src_var).pack(fill="x", ipady=6, pady=(3, 0))
        tk.Label(self.src_area, text=f"e.g. {placeholder}", font=FS,
                 bg=P["bg"], fg=P["muted"]).pack(anchor="w", pady=(2, 0))

    def _build_file(self):
        self._clear_src()
        tk.Label(self.src_area, text="Video File Path", font=FH3,
                 bg=P["bg"], fg=P["text"]).pack(anchor="w")
        row = tk.Frame(self.src_area, bg=P["bg"])
        row.pack(fill="x", pady=(3, 0))
        self.src_var = tk.StringVar()
        self._entry(row, self.src_var).pack(side="left", fill="x", expand=True, ipady=6)
        btn(row, "Browse…", self._browse, color=P["blue"], width=9
            ).pack(side="left", padx=(8, 0))

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")]
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
        raw = self.src_var.get().strip()
        road = self.road_var.get().strip()
        stype = self.src_type.get()
        if not road:
            messagebox.showwarning("Missing", "Please enter a Road / Location Name.", parent=self)
            return
        if stype == "webcam":
            if not raw.isdigit():
                messagebox.showwarning("Invalid", "Camera index must be a whole number (0, 1, 2…).", parent=self)
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
        self.result = dict(source=source, road_name=road, routes=self.routes,
                           confidence=self.conf_var.get(), source_type=stype)
        self.destroy()


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
        self.configure(bg=P["bg"])
        self.grab_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 250
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 210
        self.geometry(f"500x420+{max(0,px)}+{max(0,py)}")
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=P["blue"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  Routes for: {self.session.road_name}",
                 font=FH2, bg=P["blue"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=14)

        tk.Label(body, text="Routes are shown in the advisory panel when congestion is Moderate or higher.",
                 font=FS, bg=P["bg"], fg=P["muted"], wraplength=460, justify="left"
                 ).pack(anchor="w", pady=(0, 10))

        row = tk.Frame(body, bg=P["bg"])
        row.pack(fill="x", pady=(0, 6))
        self.ev = tk.StringVar()
        tk.Entry(row, textvariable=self.ev, font=FB, bg=P["input"], fg=P["text"],
                 insertbackground=P["text"], relief="flat",
                 highlightbackground=P["border"], highlightthickness=1
                 ).pack(side="left", fill="x", expand=True, ipady=7)
        btn(row, "+ Add", self._add, color=P["accent2"], width=7
            ).pack(side="left", padx=(8, 0))

        # Hint
        tk.Label(body,
                 text='Tip: Be descriptive — e.g. "Via Ring Road North → SH-12 South (+4 min, avoids junction)"',
                 font=FS, bg=P["bg"], fg=P["muted"], wraplength=460, justify="left"
                 ).pack(anchor="w", pady=(0, 6))

        lf = tk.Frame(body, bg=P["input"],
                      highlightbackground=P["border"], highlightthickness=1)
        lf.pack(fill="both", expand=True, pady=(0, 8))
        self.lb = tk.Listbox(lf, font=FB, bg=P["input"], fg=P["text"],
                              selectbackground=P["accent2"], relief="flat",
                              activestyle="none")
        self.lb.pack(fill="both", expand=True, padx=4, pady=4)
        for r in self.session.routes:
            self.lb.insert(tk.END, f"  {r}")

        br = tk.Frame(body, bg=P["bg"])
        br.pack(fill="x")
        btn(br, "Remove Selected", self._remove, color=P["red"], width=16).pack(side="left")
        btn(br, "Done", self.destroy, color=P["accent2"], width=8).pack(side="right")

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
        self.geometry("540x280")
        self.resizable(False, False)
        self.configure(bg=P["bg"])
        self.grab_set()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 270
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 140
        self.geometry(f"540x280+{max(0,px)}+{max(0,py)}")

        hdr = tk.Frame(self, bg=P["blue"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  PostgreSQL Database Settings",
                 font=FH2, bg=P["blue"], fg="white").pack(side="left", padx=12)

        body = tk.Frame(self, bg=P["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=16)

        tk.Label(body, text="Connection URL", font=FH3, bg=P["bg"], fg=P["text"]
                 ).pack(anchor="w")
        tk.Label(body, text="Format:  postgresql://user:password@host:port/dbname",
                 font=FS, bg=P["bg"], fg=P["muted"]).pack(anchor="w")
        self.url_var = tk.StringVar(value=current_url)
        tk.Entry(body, textvariable=self.url_var, font=FM, bg=P["input"],
                 fg=P["text"], insertbackground=P["text"], relief="flat",
                 highlightbackground=P["border"], highlightthickness=1
                 ).pack(fill="x", ipady=7, pady=(4, 16))

        tk.Label(body,
                 text="The system creates tables automatically on first connect.\nNo sample data is inserted.",
                 font=FS, bg=P["bg"], fg=P["muted"], justify="left").pack(anchor="w")

        br = tk.Frame(body, bg=P["bg"])
        br.pack(fill="x", pady=(14, 0))
        btn(br, "Cancel", self.destroy, color=P["card"], width=10).pack(side="right", padx=(8, 0))
        btn(br, "Connect", self._confirm, color=P["blue"], width=10).pack(side="right")

    def _confirm(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Missing", "Please enter a connection URL.", parent=self)
            return
        self.result = url
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA SIDEBAR CARD
# ═════════════════════════════════════════════════════════════════════════════

class CameraCard(tk.Frame):
    _STATUS_COLOR = {
        "idle": P["muted"], "running": P["accent"],
        "paused": P["amber"], "stopped": P["red"],
    }

    def __init__(self, parent, session: CameraSession, on_select, on_remove):
        super().__init__(parent, bg=P["card"],
                         highlightbackground=P["border"], highlightthickness=1,
                         cursor="hand2")
        self.session   = session
        self.on_select = on_select
        self.on_remove = on_remove
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=P["card"])
        top.pack(fill="x", padx=10, pady=(8, 2))

        self.dot = tk.Label(top, text="●", font=("Segoe UI", 10),
                            bg=P["card"], fg=self._STATUS_COLOR["idle"])
        self.dot.pack(side="left")
        tk.Label(top, text=self.session.road_name, font=FH3,
                 bg=P["card"], fg=P["text"]).pack(side="left", padx=(4, 0))

        rm = tk.Label(top, text="✕", font=("Segoe UI", 9),
                      bg=P["card"], fg=P["muted"], cursor="hand2")
        rm.pack(side="right")
        rm.bind("<Button-1>", lambda e: self.on_remove(self.session.cam_id))

        src = str(self.session.source)
        if len(src) > 36:
            src = src[:34] + "…"
        tk.Label(self, text=src, font=FS, bg=P["card"],
                 fg=P["muted"]).pack(anchor="w", padx=10, pady=(0, 7))

        self._bind_all()

    def _bind_all(self):
        for w in [self] + self.winfo_children():
            try:
                w.bind("<Button-1>", lambda e: self.on_select(self.session.cam_id))
            except Exception:
                pass

    def set_selected(self, sel: bool):
        bg = P["hover"] if sel else P["card"]
        hb = P["accent"] if sel else P["border"]
        self.config(bg=bg, highlightbackground=hb)
        for w in self.winfo_children():
            try:
                w.config(bg=bg)
            except Exception:
                pass

    def refresh(self):
        self.dot.config(fg=self._STATUS_COLOR.get(self.session.status, P["muted"]))


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING THREAD
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_youtube(url: str) -> Optional[str]:
    if not YTDLP_OK:
        return None
    try:
        ydl_opts = {
            'format': 'best[protocol^=m3u8]/best[protocol^=http]/best',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # For live streams, URL is inside formats list
            if info.get('is_live') or info.get('live_status') == 'is_live':
                formats = info.get('formats', [])
                for f in reversed(formats):
                    if f.get('protocol', '').startswith('m3u8') and f.get('url'):
                        return f['url']
                for f in reversed(formats):
                    if f.get('url'):
                        return f['url']

            return info.get('url') or info.get('manifest_url')

    except Exception as ex:
        _logger.error(f'yt-dlp error: {ex}')
        return None


def run_session(session: CameraSession, db, log_fn):
    """
    Background thread — opens camera source, runs TrafficAnalyzer per frame,
    pushes annotated PIL images to session.frame_q, writes stats to session.stats,
    and persists readings to the DB every 30 seconds.
    """
    source = session.source

    # Resolve YouTube URL to direct stream
    if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
        log_fn(f"[{session.road_name}] Resolving YouTube stream…")
        resolved = _resolve_youtube(source)
        if not resolved:
            log_fn(f"[{session.road_name}] ERROR: Could not resolve YouTube stream. "
                   f"Install yt-dlp:  pip install yt-dlp")
            session.status = "stopped"
            return
        source = resolved
        log_fn(f"[{session.road_name}] YouTube stream resolved OK.")

    if not CV2_OK:
        log_fn(f"[{session.road_name}] ERROR: OpenCV is not installed. "
               f"Run:  pip install opencv-python")
        session.status = "stopped"
        return

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_fn(f"[{session.road_name}] ERROR: Cannot open source: {source}")
        session.status = "stopped"
        return

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    log_fn(f"[{session.road_name}] Camera open — {w}×{h} @ {cam_fps:.0f} fps")

    # Load TrafficAnalyzer (YOLO model)
    analyzer = None
    if ANALYZER_OK:
        try:
            analyzer = TrafficAnalyzer(confidence=session.confidence)
            log_fn(f"[{session.road_name}] YOLO model loaded.")
        except Exception as ex:
            log_fn(f"[{session.road_name}] WARNING: Could not load analyzer: {ex}")
            log_fn(f"[{session.road_name}] Running in preview-only mode (no detection).")

    frame_skip    = 2        # process every Nth frame
    frame_idx     = 0
    last_db_write = time.time()
    DB_INTERVAL   = 30       # seconds between DB writes

    while not session.stop_ev.is_set():
        if session.pause_ev.is_set():
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            stype = session.source_type
            if stype in ("rtsp", "youtube"):
                log_fn(f"[{session.road_name}] Stream lost — retrying in 3 s…")
                time.sleep(3)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue
            else:
                # Video file finished
                log_fn(f"[{session.road_name}] End of video file.")
                break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        # Run detection
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

        # Push to GUI frame queue
        if PIL_OK:
            try:
                rgb  = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img  = Image.fromarray(rgb)
                img.thumbnail((820, 460), Image.LANCZOS)
                session.frame_q.put_nowait(img)
            except queue.Full:
                pass

        # DB write
        if db and (time.time() - last_db_write) >= DB_INTERVAL:
            try:
                db.insert_reading(
                    road_name=session.road_name,
                    vehicle_count=int(session.stats.get("total_count", 0)),
                    congestion_score=int(session.stats.get("congestion_score", 0)),
                    congestion_level=str(session.stats.get("congestion_level", "low")),
                    avg_speed=float(session.stats.get("avg_speed_kmh", 0.0)),
                    vehicle_types=dict(session.stats.get("vehicle_type_counts", {})),
                    timestamp=datetime.now(),
                )
                log_fn(f"[{session.road_name}] DB write — {session.stats.get('total_count', 0)} vehicles")
            except Exception as ex:
                log_fn(f"[{session.road_name}] DB write error: {ex}")
            last_db_write = time.time()

    cap.release()
    log_fn(f"[{session.road_name}] Session ended.")
    session.status = "stopped"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Smart Traffic Analysis System")
        self.geometry("1380x820")
        self.minsize(1100, 700)
        self.configure(bg=P["bg"])
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            self.iconbitmap("traffic_icon.ico")
        except Exception:
            pass

        # State
        self.sessions:     dict[int, CameraSession] = {}
        self.cam_cards:    dict[int, CameraCard]    = {}
        self.next_id       = 1
        self.active_id: Optional[int] = None
        self.db            = None
        self.db_url        = os.environ.get(
            "DATABASE_URL",
            "postgresql://traffic_user:traffic_pass@localhost:5432/traffic_db"
        )

        self._connect_db()
        self._build()
        self._ui_loop_preview()
        self._ui_loop_stats()
        self._tick()

        # Startup log
        self.log("Smart Traffic Analysis System — ready.")
        self._check_deps()

    # ── Dependency warnings ───────────────────────────────────────────────────
    def _check_deps(self):
        missing = []
        if not CV2_OK:
            missing.append("opencv-python  (camera + detection)")
        if not PIL_OK:
            missing.append("Pillow  (video preview)")
        if not MPL_OK:
            missing.append("matplotlib  (charts + PDF report)")
        if not DB_MOD_OK:
            missing.append("psycopg2-binary  (database)")
        if not YTDLP_OK:
            self.log("INFO: yt-dlp not installed — YouTube Live sources will not work.  pip install yt-dlp")
        if missing:
            for m in missing:
                self.log(f"WARNING: Missing package — {m}")
            self.log("Run:  pip install -r requirements_gui.txt")

    # ── Database ──────────────────────────────────────────────────────────────
    def _connect_db(self):
        if not DB_MOD_OK:
            return
        try:
            self.db = DatabaseManager(self.db_url)
            self.db.initialize()
            _logger.info("DB connected.")
        except Exception as ex:
            self.db = None
            _logger.warning(f"DB not connected: {ex}")

    def _open_db_settings(self):
        dlg = DBSettingsDialog(self, self.db_url)
        self.wait_window(dlg)
        if dlg.result:
            self.db_url = dlg.result
            self._connect_db()
            connected = self.db is not None
            self.db_status_lbl.config(
                text="● DB Connected" if connected else "○ DB Disconnected",
                fg=P["accent"] if connected else P["amber"]
            )
            self.log(f"Database: {'connected' if connected else 'failed — check URL'}")

    # ── UI construction ───────────────────────────────────────────────────────
    def _build(self):
        self._build_topbar()
        main = tk.Frame(self, bg=P["bg"])
        main.pack(fill="both", expand=True)
        self._build_sidebar(main)
        self._build_content(main)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=P["panel"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=P["accent"], width=4, height=52).pack(side="left")
        tk.Label(bar, text="  Smart Traffic Analysis System",
                 font=FT, bg=P["panel"], fg=P["text"]).pack(side="left")

        # DB button
        self.db_status_lbl = tk.Label(
            bar,
            text="● DB Connected" if self.db else "○ DB Disconnected",
            font=FS, bg=P["panel"],
            fg=P["accent"] if self.db else P["amber"],
            cursor="hand2"
        )
        self.db_status_lbl.pack(side="right", padx=(0, 16))
        self.db_status_lbl.bind("<Button-1>", lambda e: self._open_db_settings())

        self.clock_lbl = tk.Label(bar, text="", font=FS,
                                   bg=P["panel"], fg=P["muted"])
        self.clock_lbl.pack(side="right", padx=(0, 16))

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=P["panel"], width=236)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        self._sb = sb

        tk.Label(sb, text="CAMERAS", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=14, pady=(12, 6))

        # Scrollable list area
        self.cam_list = tk.Frame(sb, bg=P["panel"])
        self.cam_list.pack(fill="both", expand=True, padx=8)

        divider(sb).pack(fill="x", pady=4)
        btn(sb, "+ Add Camera", self._add_camera, color=P["accent2"], width=26
            ).pack(padx=10, pady=(4, 8), fill="x")

        # Camera controls
        tk.Label(sb, text="SELECTED CAMERA", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=14, pady=(6, 4))

        c1 = tk.Frame(sb, bg=P["panel"])
        c1.pack(fill="x", padx=8, pady=(0, 4))
        btn(c1, "▶ Start",  self._start,  color=P["accent2"], width=9).pack(side="left", padx=(0, 4))
        btn(c1, "⏸ Pause", self._pause,  color=P["amber"],   width=9).pack(side="left")

        c2 = tk.Frame(sb, bg=P["panel"])
        c2.pack(fill="x", padx=8, pady=(0, 4))
        btn(c2, "⏹ Stop",  self._stop,  color=P["red"],  width=9).pack(side="left", padx=(0, 4))
        btn(c2, "✎ Routes", self._edit_routes, color=P["blue"], width=9).pack(side="left")

        divider(sb).pack(fill="x", pady=6)
        btn(sb, "Export PDF Report", self._export_pdf, color=P["card"], width=26
            ).pack(padx=10, pady=(0, 10), fill="x")

    def _build_content(self, parent):
        ct = tk.Frame(parent, bg=P["bg"])
        ct.pack(side="left", fill="both", expand=True)

        # Top: video + right panel
        top = tk.Frame(ct, bg=P["bg"])
        top.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self._build_video_panel(top)
        self._build_right_panel(top)

        # Stats bar
        self._build_stats(ct)

        # Bottom: chart + log
        self._build_bottom(ct)

    def _build_video_panel(self, parent):
        vf = tk.Frame(parent, bg=P["panel"],
                      highlightbackground=P["border"], highlightthickness=1)
        vf.pack(side="left", fill="both", expand=True)

        # Header
        hdr = tk.Frame(vf, bg=P["panel"], height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=P["accent"], width=3, height=32).pack(side="left")
        self.vid_title = tk.Label(hdr, text="No camera selected",
                                   font=FH3, bg=P["panel"], fg=P["text"])
        self.vid_title.pack(side="left", padx=10)
        self.vid_status = tk.Label(hdr, text="● IDLE", font=FS,
                                    bg=P["panel"], fg=P["muted"])
        self.vid_status.pack(side="right", padx=10)

        # Video display
        self.video_lbl = tk.Label(vf, bg="#050A0D",
                                   text="Select a camera and press  ▶ Start",
                                   font=FB, fg=P["muted"])
        self.video_lbl.pack(fill="both", expand=True)

    def _build_right_panel(self, parent):
        rp = tk.Frame(parent, bg=P["panel"], width=278,
                      highlightbackground=P["border"], highlightthickness=1)
        rp.pack(side="left", fill="y", padx=(8, 0))
        rp.pack_propagate(False)

        tk.Label(rp, text="ROUTE ADVISORY", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=12, pady=(10, 4))

        # Congestion badge
        br = tk.Frame(rp, bg=P["panel"])
        br.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(br, text="Congestion:", font=FS,
                 bg=P["panel"], fg=P["muted"]).pack(side="left")
        self.cong_badge = tk.Label(br, text=" LOW ", font=("Segoe UI", 9, "bold"),
                                    bg="#1A4A1A", fg=P["accent"], padx=6, pady=2)
        self.cong_badge.pack(side="left", padx=(8, 0))

        # Advisory text
        self.route_txt = tk.Text(rp, font=FB, height=6, bg=P["input"],
                                  fg=P["text"], relief="flat", wrap="word",
                                  state="disabled", padx=10, pady=8)
        self.route_txt.pack(fill="x", padx=12, pady=(0, 8))

        # Configured routes list
        tk.Label(rp, text="CONFIGURED ROUTES", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=12, pady=(4, 4))
        rlf = tk.Frame(rp, bg=P["input"],
                       highlightbackground=P["border"], highlightthickness=1)
        rlf.pack(fill="x", padx=12, pady=(0, 8))
        self.routes_lb = tk.Listbox(rlf, font=FB, bg=P["input"], fg=P["text"],
                                     selectbackground=P["accent2"], relief="flat",
                                     height=4, activestyle="none")
        self.routes_lb.pack(fill="x", padx=4, pady=4)

        # Vehicle type breakdown
        tk.Label(rp, text="VEHICLE BREAKDOWN", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=12, pady=(4, 4))

        self._vbars = {}
        vtf = tk.Frame(rp, bg=P["panel"])
        vtf.pack(fill="x", padx=12)
        colors = {"car": P["accent"], "motorcycle": P["blue"],
                  "truck": P["amber"], "bus": P["red"], "bicycle": P["muted"]}
        for vt in ["car", "motorcycle", "truck", "bus", "bicycle"]:
            row = tk.Frame(vtf, bg=P["panel"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=vt.capitalize(), font=FS, width=11, anchor="w",
                     bg=P["panel"], fg=P["muted"]).pack(side="left")
            bg_f = tk.Frame(row, bg=P["input"], height=10)
            bg_f.pack(side="left", fill="x", expand=True)
            fill = tk.Frame(bg_f, bg=colors[vt], height=10, width=0)
            fill.place(x=0, y=0, relheight=1)
            cnt = tk.Label(row, text="0", font=FS, width=5, anchor="e",
                           bg=P["panel"], fg=P["text"])
            cnt.pack(side="left")
            self._vbars[vt] = (bg_f, fill, cnt)

    def _build_stats(self, parent):
        sf = tk.Frame(parent, bg=P["panel"], height=110)
        sf.pack(fill="x", padx=8, pady=(6, 0))
        sf.pack_propagate(False)

        self._stat_vals = {}
        specs = [
            ("Total Vehicles",   "total_count",     P["accent"], "#"),
            ("Live Density",     "current_density", P["blue"],   "vehicles"),
            ("Congestion Index", "congestion_score",P["amber"],  "/ 100"),
            ("Avg Speed",        "avg_speed_kmh",   P["accent"], "km/h"),
            ("Processing FPS",   "fps",             P["blue"],   "fps"),
        ]
        for i, (label, key, accent, unit) in enumerate(specs):
            card = tk.Frame(sf, bg=P["card"],
                            highlightbackground=P["border"], highlightthickness=1)
            card.pack(side="left", fill="both", expand=True,
                      padx=(0 if i > 0 else 8, 8), pady=8)
            tk.Frame(card, bg=accent, height=3).pack(fill="x")
            tk.Label(card, text=label, font=FS, bg=P["card"],
                     fg=P["muted"]).pack(pady=(6, 0))
            v = tk.Label(card, text="—", font=FST, bg=P["card"], fg=P["text"])
            v.pack()
            tk.Label(card, text=unit, font=FS, bg=P["card"],
                     fg=P["muted"]).pack(pady=(0, 6))
            self._stat_vals[key] = v

    def _build_bottom(self, parent):
        bot = tk.Frame(parent, bg=P["bg"])
        bot.pack(fill="x", padx=8, pady=(6, 8))

        # Chart
        cf = tk.Frame(bot, bg=P["panel"],
                      highlightbackground=P["border"], highlightthickness=1)
        cf.pack(side="left", fill="both", expand=True)
        tk.Label(cf, text="VEHICLE TYPE DISTRIBUTION", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=10, pady=(8, 0))
        self._chart_host = tk.Frame(cf, bg=P["panel"], height=140)
        self._chart_host.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self._chart_host.pack_propagate(False)
        self._draw_chart({})

        # Log
        lf = tk.Frame(bot, bg=P["panel"],
                      highlightbackground=P["border"], highlightthickness=1)
        lf.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(lf, text="SYSTEM LOG", font=("Segoe UI", 9, "bold"),
                 bg=P["panel"], fg=P["muted"]).pack(anchor="w", padx=10, pady=(8, 0))
        inner = tk.Frame(lf, bg=P["input"])
        inner.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.log_txt = tk.Text(inner, font=FM, height=8, bg=P["input"],
                                fg=P["muted"], relief="flat", state="disabled",
                                wrap="word", padx=6, pady=4)
        sb = tk.Scrollbar(inner, command=self.log_txt.yview,
                           troughcolor=P["input"], bg=P["border"])
        sb.pack(side="right", fill="y")
        self.log_txt.config(yscrollcommand=sb.set)
        self.log_txt.pack(fill="both", expand=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    def _draw_chart(self, vtypes: dict):
        if not MPL_OK:
            return
        for w in self._chart_host.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(4.5, 1.5), facecolor=P["panel"])
        ax.set_facecolor(P["panel"])
        if vtypes:
            labels = [k.capitalize() for k, v in vtypes.items() if v and int(v) > 0]
            vals   = [int(v) for v in vtypes.values() if v and int(v) > 0]
            colors = [P["accent"], P["blue"], P["amber"], P["red"], P["muted"]]
            if labels:
                bars = ax.barh(labels, vals, color=colors[:len(labels)], height=0.5)
                ax.set_xlim(0, max(vals) * 1.3 or 1)
                for bar, v in zip(bars, vals):
                    ax.text(v + max(vals) * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            str(v), va="center", fontsize=8, color=P["text"])
                ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "No data yet — start a camera session",
                    ha="center", va="center", color=P["muted"],
                    fontsize=9, transform=ax.transAxes)
        ax.axis("off") if not vtypes else None
        ax.tick_params(colors=P["muted"], labelsize=8)
        ax.spines[:].set_color(P["border"]) if vtypes else None
        fig.tight_layout(pad=0.3)
        canvas = FigureCanvasTkAgg(fig, master=self._chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

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
            cam_id=self.next_id, source=r["source"],
            road_name=r["road_name"], routes=r["routes"],
            confidence=r["confidence"], source_type=r["source_type"]
        )
        self.sessions[self.next_id] = session
        card = CameraCard(self.cam_list, session,
                          on_select=self._select,
                          on_remove=self._remove)
        card.pack(fill="x", pady=(0, 6))
        self.cam_cards[self.next_id] = card
        self.next_id += 1
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
        self.log(f"Camera removed: {s.road_name}")

    def _refresh_routes_lb(self, s: CameraSession):
        self.routes_lb.delete(0, tk.END)
        for r in s.routes:
            self.routes_lb.insert(tk.END, f"  {r}")

    def _refresh_status_display(self, s: CameraSession):
        colors = {"idle": P["muted"], "running": P["accent"],
                  "paused": P["amber"], "stopped": P["red"]}
        labels = {"idle": "● IDLE", "running": "● LIVE",
                  "paused": "⏸ PAUSED", "stopped": "■ STOPPED"}
        self.vid_status.config(
            text=labels.get(s.status, "● IDLE"),
            fg=colors.get(s.status, P["muted"])
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
            self.log(f"{s.road_name} is already running.")
            return
        if s.status == "paused":
            s.pause_ev.clear()
            s.status = "running"
            self._refresh_status_display(s)
            self.cam_cards[s.cam_id].refresh()
            self.log(f"{s.road_name} — resumed.")
            return
        # Fresh start
        s.stop_ev.clear()
        s.pause_ev.clear()
        s.status = "running"
        t = threading.Thread(target=run_session,
                             args=(s, self.db, self.log), daemon=True)
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
        self.log(f"{s.road_name} — routes updated ({len(s.routes)} routes).")

    # ── PDF report ────────────────────────────────────────────────────────────
    def _export_pdf(self):
        if not REPORT_OK:
            messagebox.showinfo("Not available",
                                "PDF export requires matplotlib.\n"
                                "Run:  pip install matplotlib")
            return
        if not self.db:
            messagebox.showinfo("No database",
                                "Connect to a PostgreSQL database first.\n"
                                "Click the DB status label in the top bar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        if not path:
            return
        self.log("Generating PDF report…")
        def _do():
            try:
                rg = ReportGenerator(self.db)
                rg.generate_pdf(path)
                self.after(0, lambda: self.log(f"PDF saved: {path}"))
                self.after(0, lambda: messagebox.showinfo("Done", f"Report saved:\n{path}"))
            except Exception as ex:
                self.after(0, lambda: self.log(f"PDF error: {ex}"))
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))
        threading.Thread(target=_do, daemon=True).start()

    # ── UI refresh loops ──────────────────────────────────────────────────────
    def _ui_loop_preview(self):
        """Pull latest frame from active session and display it — ~30 Hz."""
        s = self.sessions.get(self.active_id)
        if s and s.status in ("running", "paused") and PIL_OK:
            try:
                img = s.frame_q.get_nowait()
                photo = ImageTk.PhotoImage(img)
                self.video_lbl.config(image=photo, text="")
                self.video_lbl.image = photo
            except queue.Empty:
                pass
        self.after(33, self._ui_loop_preview)

    def _ui_loop_stats(self):
        """Refresh stat cards, badges, bars, and chart every 500 ms."""
        s = self.sessions.get(self.active_id)
        if s and s.status in ("running", "paused"):
            st = s.stats

            # Stat cards
            self._stat_vals["total_count"    ].config(text=str(st.get("total_count", 0)))
            self._stat_vals["current_density"].config(text=str(st.get("current_density", 0)))
            self._stat_vals["congestion_score"].config(text=str(st.get("congestion_score", 0)))
            speed = st.get("avg_speed_kmh", 0.0)
            self._stat_vals["avg_speed_kmh"  ].config(text=f"{speed:.1f}")
            fps = st.get("fps", 0.0)
            self._stat_vals["fps"            ].config(text=f"{fps:.1f}")

            # Congestion badge
            lvl = st.get("congestion_level", "low")
            badge_map = {
                "low":      ("#1A4A1A", P["accent"]),
                "moderate": ("#3D2E00", P["amber"]),
                "high":     ("#3D1500", "#F0883E"),
                "severe":   ("#3D0000", P["red"]),
            }
            bg, fg = badge_map.get(lvl, ("#1A4A1A", P["accent"]))
            self.cong_badge.config(text=f" {lvl.upper()} ", bg=bg, fg=fg)

            # Route advisory text — uses manually configured route names
            routes = s.routes
            if lvl == "low":
                msg = "Traffic is flowing normally.\nNo route diversion needed."
            elif lvl == "moderate":
                msg = "Minor delays detected.\n"
                if routes:
                    msg += "Consider:\n" + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(routes[:2]))
                else:
                    msg += "No alternative routes configured for this camera.\nClick  ✎ Routes  to add some."
            else:  # high / severe
                msg = f"{'Heavy' if lvl == 'high' else 'SEVERE'} congestion detected!\n\n"
                if routes:
                    msg += "Recommended alternatives:\n"
                    msg += "\n".join(f"  {i+1}. {r}" for i, r in enumerate(routes))
                else:
                    msg += "No routes configured.\nClick  ✎ Routes  to add alternatives."
            self.route_txt.config(state="normal")
            self.route_txt.delete("1.0", tk.END)
            self.route_txt.insert(tk.END, msg)
            self.route_txt.config(state="disabled")

            # Vehicle type bars
            vtypes = st.get("vehicle_type_counts", {})
            total_v = max(1, sum(int(v) for v in vtypes.values() if v))
            for vt, (bg_f, fill_f, cnt_l) in self._vbars.items():
                count = int(vtypes.get(vt, 0))
                cnt_l.config(text=str(count))
                pct = count / total_v
                bg_f.update_idletasks()
                w = int(bg_f.winfo_width() * pct)
                fill_f.place(x=0, y=0, width=max(0, w), relheight=1)

            # Chart — refresh every 10 calls (~5 s)
            self._chart_tick = getattr(self, "_chart_tick", 0) + 1
            if self._chart_tick % 10 == 0:
                self._draw_chart(vtypes)

        self.after(500, self._ui_loop_stats)

    # ── Clock ─────────────────────────────────────────────────────────────────
    def _tick(self):
        self.clock_lbl.config(text=datetime.now().strftime("%d %b %Y  %H:%M:%S"))
        self.after(1000, self._tick)

    # ── Close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        for s in self.sessions.values():
            s.stop_ev.set()
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
