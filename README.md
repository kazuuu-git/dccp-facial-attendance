# dccp-facial-attendance
AI-Driven Facial Recognition Attendance Logging System for DCCP Employees

## Quick start (Windows)

Requirements: Windows, a webcam, an internet connection for the first setup, and
[Python 3.11](https://www.python.org/downloads/) (tick "Add python.exe to PATH").
Newer Python versions may not support all packages.

1. Clone the repository and open the project folder.
2. Double-click **`START FIRST.bat`** once. It creates the `.venv` environment,
   installs the packages and loads the AI models. The first run downloads several
   hundred MB and can take 5–15 minutes. It is safe to run again.
3. Start the system:
   - **`dashboard.bat`** – HR dashboard in your web browser (enrollment, logs,
     personnel, export).
   - **`attendance.bat`** – webcam attendance scanner (records attendance and
     blocked attempts). Press `q` in the camera window to stop.

If setup has not been completed, each launcher runs `START FIRST.bat` automatically.
Both launchers use the local test database `data\attendance-webcam-test.db`, which is
created on first use and is never committed.

To check liveness without recording anything (diagnostics mode), run from the
project folder:

```powershell
.\.venv\Scripts\python.exe app_scanner.py --diagnostics
```
