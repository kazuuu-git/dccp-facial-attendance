@echo off
setlocal
title DCCP Dashboard
cd /d "%~dp0"

set "DCCP_DB_PATH=%~dp0data\attendance-webcam-test.db"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project Python environment was not found.
    echo Keep this launcher inside the project folder next to .venv.
    pause
    exit /b 1
)

echo Starting DCCP HR DASHBOARD
echo Database: %DCCP_DB_PATH%
echo The dashboard opens in your web browser.
echo Close this window or press Ctrl+C to stop it.
echo.

".venv\Scripts\python.exe" -m streamlit run "app_dashboard.py"

set "RESULT=%ERRORLEVEL%"
echo.
echo Dashboard stopped. Exit code: %RESULT%
pause
exit /b %RESULT%
