@echo off
title Smart Traffic Analysis System — Setup
echo.
echo  Smart Traffic Analysis System — First-Time Setup
echo  ==================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  Python found:
python --version
echo.

echo  Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo.

echo  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

echo  Installing dependencies...
pip install -r requirements_gui.txt
python -m pip install -U yt-dlp
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python
pip install onnxruntime-gpu
echo.

if not exist "models" mkdir models
if not exist "output" mkdir output
echo  Folders created: models\  output\
echo.

echo  ==================================================
echo  Setup complete!
echo  Double-click run.bat to start the application.
echo  ==================================================
echo.
pause
