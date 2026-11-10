"""
Integrated System Launcher: DCCP AI Facial Attendance
Runs both the Streamlit HR Admin Dashboard and Live Camera Scanner simultaneously.
When the scanner is closed ('q' key), the dashboard server terminates cleanly.
"""

import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    python_exe = sys.executable

    print("=" * 65)
    print("  DCCP AI FACIAL ATTENDANCE SYSTEM - ALL-IN-ONE RUNNER")
    print("=" * 65)
    print("[1/2] Starting Streamlit HR Dashboard in the background...")

    # Start Streamlit dashboard as a child process
    dashboard_process = subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", "app_dashboard.py"],
        cwd=PROJECT_ROOT,
    )

    # Allow Streamlit a moment to initialize its web server and browser tab
    time.sleep(3)

    print("[2/2] Launching Live Camera Scanner window...")
    print("      (Press 'q' in the camera window to stop both systems)")
    print("=" * 65)

    try:
        # Run camera scanner in the foreground
        scanner_process = subprocess.run(
            [python_exe, "app_scanner.py"],
            cwd=PROJECT_ROOT,
        )
    finally:
        print("\n[SHUTDOWN] Stopping Streamlit dashboard server...")
        try:
            dashboard_process.terminate()
            dashboard_process.wait(timeout=3)
        except Exception:
            dashboard_process.kill()
        print("[SHUTDOWN] System closed cleanly.")


if __name__ == "__main__":
    main()
