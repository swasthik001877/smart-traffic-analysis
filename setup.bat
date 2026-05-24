@echo off
REM Smart Traffic Analysis System — Setup for Ryzen 5 5600X + RTX 3050
REM Installs PyTorch CUDA 12.1 + Ultralytics YOLO11

echo ============================================
echo  Smart Traffic Analysis System — Setup
echo  Optimised for RTX 3050 + CUDA 12.1
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause & exit /b 1
)
python --version
echo.

python -m pip install --upgrade pip

echo [1/6] Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/6] Installing Ultralytics YOLO11...
pip install ultralytics

echo.
echo [3/6] Fixing urllib3 version conflict (requests warning)...
pip install "urllib3>=1.26.0,<2.0.0" --upgrade

echo.
echo [4/6] Installing GUI dependencies...
pip install Pillow matplotlib python-dotenv

echo.
echo [5/6] Installing database + extras...
pip install psycopg2-binary yt-dlp winotify tqdm

echo.
echo [6/6] Checking opencv-contrib-python...
pip show opencv-contrib-python >nul 2>&1
if errorlevel 1 (
    pip uninstall opencv-python opencv-python-headless -y 2>nul
    pip install opencv-contrib-python
) else (
    echo opencv-contrib-python already present, skipping.
)

echo.
echo ============================================
echo  Verifying CUDA installation...
echo ============================================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo  ABOUT TENSORRT ON WINDOWS:
echo  TensorRT is NOT installable via pip on Windows.
echo  Your app will use YOLO11 + PyTorch FP16 instead,
echo  which gives ~80-130 FPS on your RTX 3050 — more
echo  than enough for real-time traffic monitoring.
echo.
echo  To enable TensorRT (optional, adds ~2x speed):
echo    1. Download TensorRT from:
echo       https://developer.nvidia.com/tensorrt
echo    2. Follow the Windows install guide
echo    3. Then tick "TensorRT" in the camera dialog
echo.
echo  Recommended camera settings for your GPU:
echo    Model:   yolo11l
echo    TensorRT: OFF (use PyTorch FP16 instead)
echo    FP16:    ON
echo.
echo  Quick start:  run.bat
echo.
pause
