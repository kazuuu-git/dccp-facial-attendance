@echo off
setlocal
title DCCP Setup - run once
cd /d "%~dp0"

echo ============================================================
echo  DCCP Facial Attendance - first-time setup
echo.
echo  Creates the project Python environment in .venv and installs
echo  the required packages, then loads the AI models once.
echo  Needs internet. The first run downloads several hundred MB
echo  and can take 5-15 minutes. It is safe to run again.
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" goto check_existing

rem ---- Find Python 3.11, the version this project is tested with ----
set "PYCMD="
py -3.11 -c "import sys" >nul 2>&1 && set "PYCMD=py -3.11"
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo ERROR: Python 3.11 was not found on this computer.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    echo Newer versions such as 3.13+ may not support all packages.
    echo Then run "START FIRST.bat" again.
    pause
    exit /b 1
)

echo [1/4] Creating the Python 3.11 environment in .venv ...
%PYCMD% -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create the .venv environment.
    pause
    exit /b 1
)
goto install

:check_existing
echo [1/4] Existing .venv found - checking its Python version ...
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: The existing .venv does not use Python 3.11.
    echo Delete the .venv folder, then run "START FIRST.bat" again.
    pause
    exit /b 1
)

:install
if exist ".venv\.dccp-setup-complete" del ".venv\.dccp-setup-complete"

echo [2/4] Installing required packages - this can take several minutes ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo ERROR: Package installation failed. Check the internet connection
    echo and the messages above, then run "START FIRST.bat" again.
    pause
    exit /b 1
)

echo [3/4] Checking that the packages load ...
".venv\Scripts\python.exe" -c "import cv2, numpy, pandas, torch, onnxruntime, insightface, streamlit"
if errorlevel 1 (
    echo ERROR: Some packages failed to load. See the messages above.
    pause
    exit /b 1
)

echo [4/4] Loading the face recognition and liveness models ...
echo       The face model downloads once, about 120 MB.
".venv\Scripts\python.exe" -c "from src.recognition.matcher import FaceRecognitionEngine; from src.liveness.anti_spoof import PassiveLivenessEngine; FaceRecognitionEngine(); PassiveLivenessEngine()"
if errorlevel 1 (
    echo ERROR: The AI models could not be loaded. See the messages above.
    pause
    exit /b 1
)

echo ok> ".venv\.dccp-setup-complete"
echo.
echo ============================================================
echo  Setup complete.
echo  Start the system with dashboard.bat or attendance.bat.
echo ============================================================
pause
exit /b 0
