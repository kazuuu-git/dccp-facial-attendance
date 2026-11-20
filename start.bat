@echo off
setlocal
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