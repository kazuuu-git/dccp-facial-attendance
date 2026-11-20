@echo off
setlocal
title DCCP Dashboard
cd /d "%~dp0"

set "DCCP_DB_PATH=%~dp0data\attendance-webcam-test.db"

if not exist ".venv\.dccp-setup-complete" (
    echo The project environment is not set up yet.
    echo Running "START FIRST.bat" now - this is a one-time setup.
    echo.
    call "%~dp0START FIRST.bat"
)
if not exist ".venv\.dccp-setup-complete" (
    echo Setup did not finish. Fix the error shown above,
    echo then run this launcher again.
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
