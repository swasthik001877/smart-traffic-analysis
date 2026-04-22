@echo off
title Smart Traffic Analysis System
echo.
echo  Smart Traffic Analysis System
echo  ==============================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo  No virtual environment found. Run setup.bat first.
    echo.
)
python -c "import sys; print(sys.executable)"
python -c "import yt_dlp; print(yt_dlp.version.__version__)"
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
python gui_app.py

if errorlevel 1 (
    echo.
    echo  Error starting application. Check the output above.
    pause
)
