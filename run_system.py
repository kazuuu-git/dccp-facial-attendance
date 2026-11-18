"""
Integrated System Launcher: DCCP AI Facial Attendance
Runs both the Streamlit HR Admin Dashboard and Live Camera Scanner simultaneously.

- When the scanner is closed normally ('q' key, exit code 0), the dashboard
  server terminates cleanly.
- If the scanner stops abnormally (non-zero exit code), the launcher reports that
  attendance scanning has stopped and keeps the dashboard running until Ctrl+C
  or until the dashboard exits on its own. The scanner is never restarted
  automatically.
- On Ctrl+C, only the child processes started by this launcher are stopped.
"""

import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DASHBOARD_STARTUP_SECONDS = 3
POLL_INTERVAL_SECONDS = 1.0
SCANNER_GRACE_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 3.0
EXIT_INTERRUPTED = 130


def _wait_for_exit(process: subprocess.Popen) -> int:
    """Waits for a child to exit, using short timeouts so Ctrl+C stays responsive."""
    while True:
        try:
            return process.wait(timeout=POLL_INTERVAL_SECONDS)
        except subprocess.TimeoutExpired:
            continue


def _stop_child(process, name: str) -> None:
    """Terminates one child started by this launcher if it is still running."""
    if process is None or process.poll() is not None:
        return
    print(f"[SHUTDOWN] Stopping {name}...")
    try:
        process.terminate()
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except Exception as err:
            print(f"[WARN] Could not confirm that {name} stopped: {err}")


def _report_abnormal_scanner_exit(code: int) -> None:
    print("=" * 65)
    print(f"[ERROR] The camera scanner stopped unexpectedly (exit code {code}).")
    print("        ATTENDANCE SCANNING IS NOT ACTIVE: no faces are being checked or logged.")
    print("        The dashboard is still running for reviewing existing records only.")
    print("        The scanner will not restart automatically.")
    print("        Press Ctrl+C to stop the dashboard.")
    print("=" * 65)


def main() -> int:
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
    scanner_process = None
    exit_code = 0

    try:
        # Allow Streamlit a moment to initialize its web server and browser tab
        time.sleep(DASHBOARD_STARTUP_SECONDS)

        print("[2/2] Launching Live Camera Scanner window...")
        print("      (Press 'q' in the camera window to stop both systems)")
        print("=" * 65)

        # Run camera scanner in the foreground
        scanner_process = subprocess.Popen([python_exe, "app_scanner.py"], cwd=PROJECT_ROOT)
        exit_code = _wait_for_exit(scanner_process)

        if exit_code == 0:
            print("\n[INFO] Scanner closed normally.")
        else:
            _report_abnormal_scanner_exit(exit_code)
            dashboard_code = _wait_for_exit(dashboard_process)
            print(f"[INFO] Dashboard exited on its own (exit code {dashboard_code}).")
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Ctrl+C received.")
        if exit_code == 0:
            exit_code = EXIT_INTERRUPTED
        # The scanner receives Ctrl+C too; give it time to release the camera itself.
        if scanner_process is not None and scanner_process.poll() is None:
            try:
                scanner_process.wait(timeout=SCANNER_GRACE_SECONDS)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                pass
    finally:
        _stop_child(scanner_process, "camera scanner")
        _stop_child(dashboard_process, "Streamlit dashboard server")
        print("[SHUTDOWN] System closed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
