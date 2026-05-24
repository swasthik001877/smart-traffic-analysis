@echo off
REM Smart Traffic Analysis System — Launch Script
REM Fixes: debug commands behind --debug flag only

setlocal

REM Parse optional --debug flag
set DEBUG_MODE=0
for %%a in (%*) do (
    if "%%a"=="--debug" set DEBUG_MODE=1
)

if "%DEBUG_MODE%"=="1" (
    echo === DEBUG MODE ===
    python --version
    python -c "import cv2; print('OpenCV:', cv2.__version__)"
    python -c "import cv2; count=cv2.cuda.getCudaEnabledDeviceCount(); print('CUDA devices:', count)"
    python -c "import yt_dlp; print('yt-dlp:', yt_dlp.version.__version__)" 2>nul || echo yt-dlp: not installed
    python -c "from dotenv import load_dotenv; print('python-dotenv: OK')" 2>nul || echo python-dotenv: not installed
    echo =================
    echo.
)

if not exist models (
    mkdir models
)

if not exist output (
    mkdir output
)

python gui_app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo Run "run.bat --debug" for diagnostics.
    pause
)
endlocal
