# Smart Traffic Analysis System — Integrated GUI

## Project structure

```
smart_traffic_complete/
├── gui_app.py                  ← Main application — double-click or run from here
├── setup.bat                   ← First-time setup (Windows)
├── run.bat                     ← Launch the GUI (Windows)
├── requirements_gui.txt        ← All Python dependencies
├── core/
│   └── traffic_analyzer.py    ← YOLO detection + centroid tracker
├── database/
│   └── db_manager.py          ← PostgreSQL (no sample data)
├── analysis/
│   └── report_generator.py    ← PDF report generator
└── utils/
    └── logger.py              ← Logging
```

## Windows quick start

1. Double-click `setup.bat`  — creates virtual environment and installs all packages
2. Double-click `run.bat`    — launches the GUI

## Manual start (any OS)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements_gui.txt
python gui_app.py
```

## Using the GUI

1. **Add a camera** — click "+ Add Camera", choose source type and enter name + routes
2. **Select it** in the sidebar, press **▶ Start**
3. YOLO weights download automatically on first run (~236 MB)
4. **Edit routes** at any time with ✎ Routes — names you type appear in the advisory panel
5. **Export PDF** — connects to PostgreSQL and generates a full report

## Database setup (PostgreSQL)

Click the DB status label in the top bar to enter your connection URL:
```
postgresql://traffic_user:traffic_pass@localhost:5432/traffic_db
```
Tables are created automatically. No sample data is inserted.
