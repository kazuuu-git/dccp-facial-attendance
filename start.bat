@echo off
setlocal
cd /d "%~dp0"

set "DCCP_DB_PATH=%~dp0data\attendance-webcam-test.db"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project Python environment was not found.
    echo Keep this launcher inside dccp-facial-attendance-test.
    pause
    exit /b 1
)

echo Starting LIVE LIVENESS DIAGNOSTICS
echo No attendance or enrollment will be recorded.
echo Database: %DCCP_DB_PATH%
echo Close other camera apps before continuing.
echo.

".venv\Scripts\python.exe" "app_scanner.py" --diagnostics

set "RESULT=%ERRORLEVEL%"
echo.
echo Scanner stopped. Exit code: %RESULT%
pause
exit /b %RESULT%