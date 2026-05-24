<div align="center">

# 🚦 Smart Traffic Analysis System

**Real-time vehicle detection, tracking, congestion scoring, and traffic analytics**
**powered by YOLO11 and ByteTracker**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00BFFF?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAyNCAyNCc+PHBhdGggZmlsbD0nd2hpdGUnIGQ9J00xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNFY3bC04IDQtOC00eicvPjwvc3ZnPg==)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)](https://microsoft.com/windows)

<br/>

![Demo](https://img.shields.io/badge/Status-Active-brightgreen)
&nbsp;
![GPU](https://img.shields.io/badge/GPU-RTX%203050%2B-76B900?logo=nvidia&logoColor=white)
&nbsp;
![Database](https://img.shields.io/badge/DB-SQLite%20%7C%20PostgreSQL-003B57?logo=sqlite&logoColor=white)

</div>

---

## Overview

Smart Traffic Analysis System is a desktop application for real-time traffic monitoring. Point it at a webcam, IP camera, RTSP stream, YouTube Live feed, or a recorded video file — it detects and tracks vehicles using YOLO11, computes congestion scores, estimates speeds, manages incidents, fires alerts, and generates detailed PDF reports, all from a single Windows GUI.

**Key highlights:**

- Detects cars, motorcycles, trucks, buses, and bicycles using YOLO11 with ByteTracker
- Congestion scoring (0–100) across four levels with desktop alerts
- Route advisory panel — shows alternative routes when congestion is Moderate or higher
- Full incident management — create, view, resolve
- Multi-page PDF reports with hourly trends, heatmaps, and per-road forecasts
- SQLite out of the box, PostgreSQL for production deployments
- Cameras persist between restarts — no re-entering sources

---

## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Camera Sources](#camera-sources)
- [YOLO Model Selection](#yolo-model-selection)
- [Hardware Recommendations](#hardware-recommendations)
- [GPU Acceleration](#gpu-acceleration)
- [Database Configuration](#database-configuration)
- [Speed Calibration](#speed-calibration)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/smart-traffic-analysis.git
cd smart-traffic-analysis

# 2. Install dependencies (handles PyTorch CUDA, Ultralytics, OpenCV)
setup.bat

# 3. Launch
run.bat
```

That's it. SQLite is used by default — no database setup needed. The YOLO11 model (~49 MB) downloads automatically the first time you start a camera session.

For diagnostic output on startup:
```bash
run.bat --debug
```

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| Python | 3.10 | 3.11 |
| CPU | Any x64 | Ryzen 5 5600X / Ryzen 7000H series |
| GPU | None (CPU mode) | NVIDIA RTX 3050+ with CUDA 12.1 |
| RAM | 8 GB | 16 GB |
| VRAM | — | 4 GB+ (8 GB for `yolo11l`) |
| Disk | 500 MB | 1 GB (models + recordings) |

> **No GPU?** The app runs in CPU mode with `yolo11n` or `yolo11s` at 4–8 FPS — suitable for offline video file analysis.

---

## Installation

### Step 1 — Install Python 3.11

Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**.

### Step 2 — Clone or download this repository

```bash
git clone https://github.com/yourusername/smart-traffic-analysis.git
cd smart-traffic-analysis
```

### Step 3 — Run setup

```bash
setup.bat
```

`setup.bat` installs the following in the correct order:

| Step | Package | Purpose |
|------|---------|---------|
| 1 | `torch` + `torchvision` (CUDA 12.1) | GPU acceleration |
| 2 | `ultralytics` | YOLO11, ByteTracker, model management |
| 3 | `urllib3<2.0` | Fix requests version conflict |
| 4 | `Pillow`, `matplotlib`, `python-dotenv` | GUI + charts + config |
| 5 | `psycopg2-binary` | PostgreSQL (optional) |
| 6 | `yt-dlp`, `winotify`, `tqdm` | YouTube streams, alerts, progress |
| 7 | `opencv-contrib-python` | Video capture + CUDA DNN backend |

> **PyTorch note:** `setup.bat` installs the CUDA 12.1 build of PyTorch directly. Installing from `requirements_gui.txt` alone gives the CPU-only build — always use `setup.bat` on a fresh install.

### Step 4 — Launch

```bash
run.bat
```

---

## Project Structure

```
smart-traffic-analysis/
│
├── gui_app.py                  ← Main application entry point
│                                  GUI, camera management, session control,
│                                  incident panel, alerts, PDF export
│
├── core/
│   └── traffic_analyzer.py    ← YOLO11 detection engine
│                                  ByteTracker integration, speed estimation,
│                                  congestion scoring, TensorRT support
│
├── database/
│   └── db_manager.py          ← Database backend
│                                  PostgreSQL + SQLite, schema creation,
│                                  all analytics queries, incident CRUD
│
├── analysis/
│   └── report_generator.py    ← PDF report generator
│                                  Matplotlib multi-page report,
│                                  dynamic road names from DB
│
├── utils/
│   └── logger.py              ← Centralised logger
│                                  Module-level caching, no duplicate handlers
│
├── models/
│   ├── yolo11l.pt             ← Downloaded automatically on first run (~49 MB)
│   ├── yolo11l_fp16.engine    ← TensorRT engine, compiled once + cached (optional)
│   └── coco.names             ← 80 COCO class names
│
├── output/                    ← PDF reports + recorded videos saved here
│
├── cameras.json               ← Camera list, auto-saved + restored on restart
├── traffic.db                 ← SQLite database (default mode)
│
├── .env                       ← Your local config (not committed to git)
├── .env.example               ← Config template
├── requirements_gui.txt       ← Python dependencies
├── setup.bat                  ← Windows installer
└── run.bat                    ← Windows launcher
```

---

## Features

### 🎥 Camera Sources
| Source | How to add |
|--------|-----------|
| Webcam | Device index — `0` for first camera, `1` for second |
| RTSP / IP Camera | Full URL e.g. `rtsp://admin:pass@192.168.1.100:554/stream` |
| YouTube Live | Paste the `youtube.com/watch?v=...` URL directly |
| Video File | Browse for `.mp4`, `.avi`, `.mov`, `.mkv` |

### 🔍 Detection & Tracking
- **YOLO11** with 10 model variants (nano → extra-large) — selectable per camera
- **ByteTracker** for persistent vehicle IDs across frames
- Detects: cars, motorcycles, trucks, buses, bicycles
- Bounding boxes, track IDs, and vehicle type drawn on live feed
- Configurable detection confidence threshold per camera

### 📊 Analysis
- Cumulative vehicle count + live frame density
- **Congestion score 0–100** with four levels: `LOW` / `MODERATE` / `HIGH` / `SEVERE`
- Average speed estimation — calibrated via pixels-per-metre per camera
- Configurable counting line position (default 55% of frame height)
- Configurable max density threshold (what vehicle count = score 100)
- All data written to database at a configurable interval (default 30 s)

### 🗺 Route Advisory
- Configure alternative routes per camera (e.g. `"Ring Road North → City Bypass (+4 min)"`)
- Advisory panel activates automatically at Moderate congestion and above
- Edit routes live without stopping a running session (`Ctrl+E`)

### 🚨 Incident Management
- Report incidents: road name, description, severity (low / medium / high / critical)
- View and resolve active incidents in real time (`Ctrl+I`)
- Incidents written to database, included in PDF reports

### 🔔 Alerts
- Audio beep when congestion reaches High or Severe
- Windows desktop toast notification via [winotify](https://github.com/versa-syahptr/winotify)
- Alert only fires on level *transition* — no repeated notifications for sustained congestion

### 🎬 Video Recording
- Optional per-camera recording of the annotated video feed to `output/`
- Toggle in Advanced Options when adding or editing a camera
- Saved as `.mp4` with codec `mp4v`

### 📄 PDF Reports
- Date-range selector — choose any window from 1 hour to 7 days (`Ctrl+P`)
- **Pages included:**
  - Title page with summary stats (total vehicles, avg speed, avg congestion, active incidents)
  - Hourly vehicle count + congestion trends — all roads overlaid
  - Vehicle type distribution — donut chart + bar chart
  - Road usage — avg congestion + avg speed per road
  - Congestion heatmap — hour × road matrix
  - Per-road congestion + speed forecast (one page per road)
- All charts use actual road names from the database — nothing hardcoded

### 💾 Session Persistence
- Camera list auto-saved to `cameras.json` after every add, edit, or remove
- Fully restored on next launch — no re-entering sources or settings

### 🗄 Database
- **SQLite** (default) — zero setup, single file, works immediately
- **PostgreSQL** — for multi-camera or multi-user production deployments
- Schema created automatically on first launch
- No sample data inserted — all data from your live sessions

---

## YOLO Model Selection

Select per camera in **Advanced Options → YOLO Model** when adding or editing a camera.

| Model | File size | mAP50-95 | Desktop RTX 3050 8 GB | Laptop RTX 3050 4 GB | CPU only |
|-------|-----------|----------|----------------------|----------------------|----------|
| `yolo11n` | 5 MB | 39.5 | ~180 FPS | ~100 FPS | ~8 FPS |
| `yolo11s` | 19 MB | 47.0 | ~120 FPS | ~70 FPS | ~4 FPS |
| `yolo11m` | 40 MB | 51.5 | ~80 FPS | **~45–60 FPS** ✅ | ~2 FPS |
| `yolo11l` | 49 MB | 53.4 | **~50–90 FPS** ✅ | ~25–35 FPS ⚠️ | — |
| `yolo11x` | 110 MB | 54.7 | ~30 FPS | ⚠️ VRAM risk | — |
| `yolov8n` → `yolov8x` | 6–131 MB | varies | alternative family | — | — |

All models download automatically from Ultralytics CDN on first use. Real-time monitoring needs ~25 FPS minimum — any GPU model above `yolo11s` exceeds this comfortably.

---

## Hardware Recommendations

### 🖥 Desktop — NVIDIA RTX 3050 (8 GB VRAM)
- **Model:** `yolo11l`
- **Mode:** PyTorch FP16 (auto-enabled)
- **FPS:** ~50–90 FPS
- Supports 2–3 simultaneous cameras

### 💻 Laptop — NVIDIA RTX 3050 (4 GB VRAM)
- **Model:** `yolo11m`
- **Mode:** PyTorch FP16 (auto-enabled)
- **FPS:** ~45–60 FPS
- The laptop RTX 3050 delivers ~4.33 TFLOPS vs the desktop's 9.1 TFLOPS. `yolo11l` will run but pushes against the 4 GB VRAM limit — `yolo11m` is the safe, stable choice.
- Keep to **one camera at a time** to avoid VRAM pressure
- FPS varies with TGP wattage (35W–80W) and thermals

### ⚙️ CPU Only (no NVIDIA GPU)
- **Model:** `yolo11n` or `yolo11s`
- **FPS:** ~4–8 FPS
- Usable for offline analysis of pre-recorded video files

---

## GPU Acceleration

### PyTorch FP16 — Recommended for Windows

Enabled automatically when an NVIDIA GPU with CUDA is detected. No configuration needed.

Expected speed on RTX 3050: **~45–130 FPS** depending on model choice — well above the 25 FPS needed for real-time monitoring.

### TensorRT FP16 — Optional (2–3× extra speed)

> ⚠️ **TensorRT is not pip-installable on Windows.** The `pip install tensorrt` package only works on Linux.

To enable on Windows:
1. Download the TensorRT SDK from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)
2. Follow the [Windows installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
3. Tick **Enable TensorRT FP16** in Advanced Options when adding a camera
4. The first session compiles the engine (~2 minutes) — it is then cached permanently in `models/`

On Linux, TensorRT installs via `pip install tensorrt` and works without the SDK.

**If TensorRT is unavailable**, the app automatically falls back to PyTorch FP16 — no crash, no error message, no YOLOv3 fallback.

---

## Database Configuration

### SQLite — Default, no setup required

Data is stored in `traffic.db` in the project folder. Works out of the box.

### PostgreSQL — For production / multi-user deployments

Create a `.env` file in the project root (copy from `.env.example`):

```env
DATABASE_URL=postgresql://username:password@localhost:5432/traffic_db
```

Or click the **DB status label** in the top bar of the running app to enter and test the URL without restarting.

Tables are created automatically on first launch. No sample data is inserted.

**Schema overview:**

```sql
traffic_readings   -- vehicle count, congestion score, speed, vehicle types per road per interval
incidents          -- reported incidents with severity and resolution timestamp
```

---

## Speed Calibration

Speed estimation uses a configurable **pixels-per-metre** scale factor set per camera in Advanced Options.

**Default:** `20 px/m` — approximately a camera mounted ~5 m above the road looking horizontally.

**To calibrate for your camera:**

1. Identify a known real-world distance visible in the frame (e.g. a standard lane width ≈ 3.5 m)
2. Measure how many pixels it spans in the video preview
3. Calculate: `pixels_per_meter = pixel_span ÷ real_metres`

**Example:** A lane appears 70 px wide on screen → `70 ÷ 3.5 = 20 px/m`

> **Note:** Speed estimates are approximate. Accuracy depends on camera angle, height, and lens. For certified speed measurements, use dedicated radar or LiDAR sensors.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Start / Resume selected camera |
| `Escape` | Stop selected camera |
| `Ctrl + N` | Add new camera |
| `Ctrl + E` | Edit routes for selected camera |
| `Ctrl + R` | Reset vehicle counts for selected camera |
| `Ctrl + I` | Open incident management panel |
| `Ctrl + P` | Export PDF report |

---

## Configuration Reference

All settings are stored in `.env`. Copy `.env.example` to get started.

```env
# Database connection
# SQLite (default — no setup):
DATABASE_URL=sqlite:///traffic.db

# PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost:5432/traffic_db

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

Per-camera settings (set in the GUI, stored in `cameras.json`):

| Setting | Default | Description |
|---------|---------|-------------|
| YOLO model | `yolo11l` | Detection model variant |
| Confidence | `0.45` | Minimum detection confidence |
| DB write interval | `30 s` | How often readings are written to the database |
| Counting line position | `0.55` | Y position of counting line as fraction of frame height |
| Pixels per metre | `20` | Speed calibration factor |
| Max density | `20` | Vehicle count that equals congestion score 100 |
| TensorRT | Off | Use TensorRT FP16 engine (requires SDK install) |
| FP16 | On | Use half-precision on GPU |
| Record video | Off | Save annotated session video to `output/` |

---

## Troubleshooting

Run `run.bat --debug` to print version and CUDA diagnostics on startup.

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Camera won't open | Wrong source index or URL | Re-check source type and value in camera settings |
| `No module named 'tensorrt'` | TensorRT SDK not installed | Normal on Windows — app falls back to PyTorch FP16 automatically |
| YouTube stream fails | yt-dlp missing or stream is private/members-only | `pip install yt-dlp`; ensure stream is public |
| CUDA not detected | CPU-only PyTorch installed | Re-run `setup.bat`; verify NVIDIA drivers are installed |
| No DB writes | Wrong connection URL | Click DB status label in top bar; check `.env` |
| PDF charts are empty | No session data in DB yet | Run at least one camera session to populate the database |
| App is slow on CPU | No GPU or CUDA not detected | Switch model to `yolo11n` or `yolo11s` |
| `urllib3` version warning | requests/urllib3 version mismatch | Re-run `setup.bat` (step 3 pins urllib3) |
| Vehicle counts seem inflated | Counting line too low / wrong position | Adjust "Counting line position" in Advanced Options |
| Speed values are way off | `pixels_per_meter` not calibrated | Measure and set the correct px/m value per camera |
| App crashes on startup | Missing dependency | Run `run.bat --debug` and check which import fails |
| YOLO model keeps re-downloading | `models/` folder missing or wrong path | Ensure you run `run.bat` from the project root directory |

---

## Known Limitations

- **Speed estimation is approximate** — requires per-camera calibration and assumes a flat, level road viewed at a consistent angle. Not suitable for legal enforcement.
- **Congestion score is density-based** — measures how many vehicles are in frame, not actual road capacity or flow rate.
- **YouTube Live reconnect latency** — after a stream interruption, reconnect takes 3–5 seconds.
- **TensorRT on Windows** — requires manual NVIDIA SDK installation; not available via pip.
- **Prediction accuracy** — the per-road forecast pages in the PDF report require at least 7 days of historical readings to produce meaningful predictions.
- **Single process** — all cameras share one Python process and one GPU. Very high camera counts may compete for VRAM.
- **No authentication** — the app has no login system; all users on the machine have full access.

---

## Roadmap

Potential features for future versions:

- [ ] 2×2 multi-camera grid view
- [ ] Wrong-way vehicle detection
- [ ] Drag-to-draw polygon counting zones
- [ ] Email / Telegram / WhatsApp alerts
- [ ] Web dashboard (Flask/FastAPI)
- [ ] REST API for external system integration
- [ ] Interactive map view with per-camera congestion pins
- [ ] Automatic DB cleanup / data retention settings
- [ ] Scheduled session start/stop per camera
- [ ] Camera calibration wizard (click two points → auto px/m)

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is for personal and educational use.

Detection is powered by [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). For commercial deployment, an [Ultralytics Enterprise License](https://www.ultralytics.com/license) is required.

---

<div align="center">

Made with Python · YOLO11 · PyTorch · OpenCV · Tkinter

</div>
