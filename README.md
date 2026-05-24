# Smart Traffic Analysis System

Real-time traffic monitoring using YOLO11 vehicle detection, multi-camera management,
PostgreSQL/SQLite logging, incident management, and PDF reporting.

Built with Python + Tkinter. Optimised for **Windows + NVIDIA GPU**.

---

## Quick Start

```
setup.bat
run.bat
```

- No database setup needed — uses SQLite (`traffic.db`) by default
- YOLO11 model (~49 MB) downloads automatically on first camera start
- Cameras are saved and restored automatically between sessions

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| Python | 3.10 | 3.11 |
| CPU | Any modern CPU | Ryzen 5 5600X or better |
| GPU | None (CPU mode) | NVIDIA RTX 3050+ with CUDA 12.1 |
| RAM | 8 GB | 16 GB |
| VRAM | — | 4 GB+ |

---

## Installation

### 1. Install Python 3.11
Download from [python.org](https://www.python.org/downloads/). Check **"Add to PATH"** during install.

### 2. Run setup
```
setup.bat
```

This installs, in order:
- PyTorch with CUDA 12.1 (GPU acceleration)
- Ultralytics (YOLO11 + ByteTracker)
- OpenCV, Pillow, Matplotlib
- psycopg2, python-dotenv, yt-dlp, winotify

### 3. Launch
```
run.bat
```

Add `--debug` for diagnostic output:
```
run.bat --debug
```

---

## Project Structure

```
smart-traffic-analysis-main/
├── gui_app.py                  Main application — GUI, camera management, session control
├── core/
│   └── traffic_analyzer.py     YOLO11 detection, ByteTracker, speed + congestion scoring
├── database/
│   └── db_manager.py           PostgreSQL + SQLite backend, all analytics queries
├── analysis/
│   └── report_generator.py     Multi-page Matplotlib PDF report generator
├── utils/
│   └── logger.py               Centralised logger with module-level caching
├── models/
│   ├── yolo11l.pt              Downloaded automatically on first run (~49 MB)
│   ├── yolo11l_fp16.engine     TensorRT engine — compiled once, cached (optional)
│   └── coco.names              80 COCO class names
├── output/                     PDF reports and recorded videos saved here
├── cameras.json                Camera list — auto-saved and restored on restart
├── .env                        Your local config (not committed to git)
├── .env.example                Config template
├── requirements_gui.txt        Python dependencies
├── setup.bat                   Windows dependency installer
└── run.bat                     Windows launcher
```

---

## Features

### Camera Sources
- **Webcam** — by device index (0, 1, 2…)
- **RTSP / IP Camera** — any `rtsp://` stream URL
- **YouTube Live** — paste the watch URL directly (requires yt-dlp)
- **Video File** — MP4, AVI, MOV, MKV

### Detection & Tracking
- **YOLO11** (default: `yolo11l`) — cars, motorcycles, trucks, buses, bicycles
- **ByteTracker** built-in — persistent vehicle IDs across frames
- **10 model variants** — YOLO11n/s/m/l/x and YOLOv8n/s/m/l/x, selectable per camera
- Vehicle type, track ID, and bounding box drawn on live feed

### Analysis
- Vehicle count (cumulative + live density)
- Congestion score 0–100 with four levels: Low / Moderate / High / Severe
- Average speed estimation (calibrated per camera via pixels-per-metre setting)
- Peak hours, hourly trends, road usage stats all written to database

### Route Advisory
- Configurable alternative routes per camera
- Advisory panel shows suggestions at Moderate congestion and above
- Routes editable live without stopping the session (`Ctrl+E`)

### Incident Management (`Ctrl+I`)
- Report incidents with road name, description, and severity level
- View and resolve active incidents in real time
- Incidents stored in database, appear in PDF reports

### Alerts
- Audio beep when congestion becomes High or Severe
- Windows desktop toast notification (via winotify)

### Video Recording
- Optional per-camera recording of annotated video to `output/`
- Enabled in Advanced Options when adding a camera

### PDF Reports (`Ctrl+P`)
- Date range selector (1 hour to 7 days)
- Pages: title + summary stats, hourly trends, vehicle distribution,
  road usage, congestion heatmap, per-road forecasts
- All charts generated from actual database data — no hardcoded road names

### Database
- **SQLite** (default) — zero setup, single file, works out of the box
- **PostgreSQL** — for multi-camera or multi-user deployments
- Schema created automatically on first launch
- No sample data inserted

### Session Persistence
- Camera list saved to `cameras.json` on every change
- Fully restored on restart — no re-entering cameras

---

## YOLO Model Selection

Set per camera in **Advanced Options → YOLO Model**.

| Model | Size | mAP | RTX 3050 FPS | Use case |
|-------|------|-----|--------------|----------|
| yolo11n | 5 MB | 39.5 | ~180 FPS | CPU or very low latency |
| yolo11s | 19 MB | 47.0 | ~120 FPS | Balanced on CPU |
| yolo11m | 40 MB | 51.5 | ~80 FPS | General purpose |
| **yolo11l** | **49 MB** | **53.4** | **~50–90 FPS** | **Recommended for RTX 3050** |
| yolo11x | 110 MB | 54.7 | ~30 FPS | Highest accuracy, GPU required |
| yolov8n–x | 6–131 MB | varies | — | Alternative proven family |

All models auto-download from Ultralytics CDN on first use.

---

## GPU Acceleration

### PyTorch FP16 (recommended for Windows)
Enabled automatically when an NVIDIA GPU is detected.
Gives **~80–130 FPS** on RTX 3050 with `yolo11l` — more than enough for real-time monitoring.
No extra setup required.

### TensorRT FP16 (optional, Linux / manual SDK)
TensorRT is **not pip-installable on Windows**. It requires the full NVIDIA TensorRT SDK:
- Download: https://developer.nvidia.com/tensorrt
- Follow the Windows install guide
- Once installed, tick **"Enable TensorRT FP16"** in Advanced Options
- First run compiles the engine (~2 min), then it is cached permanently
- Expected speed: ~130–150 FPS on RTX 3050

If TensorRT is unavailable, the app automatically falls back to PyTorch FP16 — no crash, no YOLOv3 fallback.

---

## Database Configuration

### SQLite (default — no setup)
Works out of the box. Data stored in `traffic.db` in the project folder.

### PostgreSQL
Create a `.env` file (copy from `.env.example`):
```
DATABASE_URL=postgresql://username:password@localhost:5432/traffic_db
```

Or click the **DB status label** in the top bar to enter the URL from inside the app.
Tables are created automatically. No sample data is inserted.

---

## Speed Calibration

Speed estimation uses a **pixels-per-metre** scale factor, configurable per camera in Advanced Options.

Default: `20 px/m` — roughly a camera mounted ~5 m above a road, looking horizontally.

To calibrate for your camera:
1. Identify a known real-world distance in the frame (e.g. a lane width ≈ 3.5 m)
2. Count how many pixels it spans on screen
3. Set: `pixels_per_meter = pixel_span ÷ real_metres`

Example: lane appears 70 px wide → `70 ÷ 3.5 = 20 px/m`

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start / Resume selected camera |
| `Escape` | Stop selected camera |
| `Ctrl+N` | Add new camera |
| `Ctrl+E` | Edit routes for selected camera |
| `Ctrl+R` | Reset vehicle counts |
| `Ctrl+I` | Open incident management panel |
| `Ctrl+P` | Export PDF report |

---

## Troubleshooting

Run `run.bat --debug` to print version and CUDA diagnostics on startup.

| Problem | Cause | Fix |
|---------|-------|-----|
| Camera won't open | Wrong source index or URL | Re-check source type and value in camera settings |
| `No module named 'tensorrt'` | TensorRT not installed | Normal on Windows — app falls back to PyTorch FP16 automatically |
| YouTube stream fails | yt-dlp missing or stream is private | `pip install yt-dlp`; ensure stream is public |
| CUDA not detected | Wrong PyTorch build | Re-run `setup.bat`; verify CUDA drivers installed |
| No DB writes | Wrong connection URL | Click DB label in top bar; check `.env` file |
| PDF charts are empty | No session data in DB | Run at least one camera session first |
| App slow on CPU | No GPU or CUDA | Use `yolo11n` or `yolo11s` for CPU mode |
| `urllib3` version warning | requests/urllib3 mismatch | Re-run `setup.bat` (step 3 fixes this) |
| Counts seem too high | Counting line position wrong | Adjust "Counting line position" in Advanced Options |
| Speed values look wrong | pixels_per_meter not calibrated | Measure and set px/m per camera (see Speed Calibration above) |

---

## Known Limitations

- Speed estimation is approximate — requires per-camera calibration for accurate readings
- Congestion score is based on object density in frame, not road capacity
- YouTube Live reconnect may take 3–5 seconds after a stream interruption
- TensorRT requires manual SDK install on Windows (pip package not available)
- PDF report prediction pages require at least 7 days of historical data for accuracy

---

## License

For personal and educational use. Ultralytics YOLO11 is licensed under
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
For commercial deployment, obtain an Ultralytics Enterprise license.
