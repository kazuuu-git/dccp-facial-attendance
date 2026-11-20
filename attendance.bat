@echo off
setlocal
title DCCP Attendance Scanner
cd /d "%~dp0"

set "DCCP_DB_PATH=%~dp0data\attendance-webcam-test.db"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project Python environment was not found.
    echo Keep this launcher inside the project folder next to .venv.
    pause
    exit /b 1
)

echo Starting DCCP ATTENDANCE SCANNER
echo Attendance and blocked attempts WILL be recorded.
echo Database: %DCCP_DB_PATH%
echo Close other camera apps, including the dashboard's Enroll page, first.
echo Press 'q' in the camera window to stop.
echo.

".venv\Scripts\python.exe" "app_scanner.py"

set "RESULT=%ERRORLEVEL%"
echo.
echo Scanner stopped. Exit code: %RESULT%
pause
exit /b %RESULT%
