"""
Milestone 3: Live Camera Attendance Scanner Pipeline
Owner: Karl Stephen V. Zubiri

Integrates:
  - Face recognition via InsightFace (src/recognition/matcher.py)
  - Passive anti-spoofing via MiniFASNetV2 (src/liveness/anti_spoof.py)
  - Persistent SQLite logging & enrollment (src/database/db_manager.py)

Tuning (environment variables):
  DCCP_PROCESS_INTERVAL   seconds between recognition attempts (default 0.25 = 4/sec)
  DCCP_MIN_FACE_WIDTH     minimum face width in pixels before recognizing (default 80)
  DCCP_ORT_THREADS        ONNX Runtime threads (default 2, read in matcher.py)
  DCCP_EMPLOYEE_LOCKOUT   cooldown before re-logging the same employee (default 60s)
  DCCP_SPOOF_COOLDOWN     cooldown between repeated spoof log alerts (default 4s)
  DCCP_MAX_READ_FAILURES  consecutive failed camera reads before the scanner stops
                          with exit code 2 (positive integer, default 30)
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.recognition.matcher import FaceRecognitionEngine

PROCESS_INTERVAL = float(os.getenv("DCCP_PROCESS_INTERVAL", "0.25"))
MIN_FACE_WIDTH = int(os.getenv("DCCP_MIN_FACE_WIDTH", "80"))
EMPLOYEE_LOCKOUT_SECONDS = float(os.getenv("DCCP_EMPLOYEE_LOCKOUT", "60.0"))
SPOOF_COOLDOWN_SECONDS = float(os.getenv("DCCP_SPOOF_COOLDOWN", "4.0"))

READY_BANNER = "Ready: Look at camera (Press 'e' to test-enroll)"
GREY = (200, 200, 200)
ORANGE = (0, 165, 255)

DEFAULT_MAX_READ_FAILURES = 30
READ_RETRY_DELAY_SECONDS = 0.1
WINDOW_TITLE = "DCCP AI Facial Attendance Scanner"

EXIT_OK = 0
EXIT_CAMERA_READ_FAILURE = 2
EXIT_CONFIG_ERROR = 3
EXIT_INTERRUPTED = 130


class CameraReadError(RuntimeError):
    """The camera stopped delivering usable frames; scanning has stopped."""


class ScannerConfigError(ValueError):
    """A scanner configuration value is invalid."""


def parse_max_read_failures(raw: Optional[str]) -> int:
    """Validates DCCP_MAX_READ_FAILURES. Unset or blank uses the default."""
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_READ_FAILURES
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value < 1:
        raise ScannerConfigError(
            f"DCCP_MAX_READ_FAILURES must be a positive integer, got {raw!r}."
        )
    return value


def _is_usable_frame(ret: bool, frame) -> bool:
    """A read counts as successful only if it returned a non-empty image array."""
    return bool(ret) and isinstance(frame, np.ndarray) and frame.ndim >= 2 and frame.size > 0


def _camera_backend_name(cap) -> str:
    """Best-effort backend name for diagnostics; never fails camera operation."""
    try:
        return str(cap.getBackendName())
    except Exception:
        return "unknown"

# Required components. There is deliberately NO fallback: an always-pass liveness
# stub or an in-memory database would let the scanner record attendance or enroll
# faces without real verification or persistence. Fail before any camera work.
try:
    from src.liveness.anti_spoof import PassiveLivenessEngine
except Exception as e:
    raise ImportError(
        "Scanner cannot start: required passive liveness module "
        f"'src.liveness.anti_spoof' failed to import ({type(e).__name__}: {e}). "
        "Install the liveness dependencies before running the scanner."
    ) from e

try:
    from src.database.db_manager import AttendanceDatabase
except Exception as e:
    raise ImportError(
        "Scanner cannot start: required database module "
        f"'src.database.db_manager' failed to import ({type(e).__name__}: {e})."
    ) from e


def _print_latency(name: str, samples_s: List[float]) -> None:
    """Prints average / p95 / worst-case latency in milliseconds."""
    if not samples_s:
        return
    ms = np.array(samples_s) * 1000.0
    print(
        f"[PERF] {name}: n={len(ms)} avg={ms.mean():.0f} ms "
        f"p95={np.percentile(ms, 95):.0f} ms max={ms.max():.0f} ms"
    )


def run_attendance_scanner(
    camera_index: int = 0,
    lockout_seconds: float = EMPLOYEE_LOCKOUT_SECONDS,
    spoof_cooldown_seconds: float = SPOOF_COOLDOWN_SECONDS,
    process_interval: float = PROCESS_INTERVAL,
    max_read_failures: Optional[int] = None,
    read_retry_delay: float = READ_RETRY_DELAY_SECONDS,
) -> None:
    """Runs the real-time DCCP employee attendance kiosk loop.

    Raises CameraReadError if the camera returns `max_read_failures` unusable
    frames in a row, and ScannerConfigError for invalid configuration.
    """
    # Validate configuration before loading models or touching the camera.
    if max_read_failures is None:
        max_read_failures = parse_max_read_failures(os.getenv("DCCP_MAX_READ_FAILURES"))
    elif isinstance(max_read_failures, bool) or not isinstance(max_read_failures, int) or max_read_failures < 1:
        raise ScannerConfigError(f"max_read_failures must be a positive integer, got {max_read_failures!r}.")

    db = AttendanceDatabase()
    liveness_engine = PassiveLivenessEngine()
    recognition_engine = FaceRecognitionEngine()

    enrolled_db = db.load_enrolled_embeddings()
    print(f"[INFO] Loaded {len(enrolled_db)} enrolled DCCP employee profile(s).")
    print(f"[INFO] Recognition interval: {process_interval:.2f}s | Min face width: {MIN_FACE_WIDTH}px | Lockout: {lockout_seconds:.0f}s")
    print("[CONTROLS] Press 'e' to quick-enroll your face for testing | Press 'q' to quit.")

    rec_times: List[float] = []
    live_times: List[float] = []

    # Open the camera ONCE. Everything after this point runs inside try/finally so
    # the camera is released and windows are closed on any exit, including Ctrl+C.
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}.")
        backend = _camera_backend_name(cap)
        print(f"[INFO] Camera opened: index {camera_index}, backend {backend}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        last_logged_time: Dict[str, float] = {}
        status_banner = READY_BANNER
        banner_color = GREY

        bbox: Optional[list] = None  # cached between recognition cycles (for drawing only)
        last_proc = 0.0
        consecutive_read_failures = 0
        frame_shape = (480, 640, 3)

        def _get_db_mtime() -> float:
            try:
                return os.path.getmtime(db.db_path) if os.path.exists(db.db_path) else 0.0
            except OSError:
                return 0.0

        last_db_mtime = _get_db_mtime()
        last_db_check = 0.0

        while True:
            ret, frame = cap.read()
            if not _is_usable_frame(ret, frame):
                consecutive_read_failures += 1
                # Cached results describe an earlier frame: drop them so nothing
                # (e.g. a previous PASS) is shown for input we did not receive, and
                # force fresh recognition on the next usable frame.
                bbox = None
                status_banner, banner_color = READY_BANNER, GREY
                last_proc = 0.0

                if consecutive_read_failures == 1:
                    print("[WARN] Camera returned no usable frame; retrying.")
                if consecutive_read_failures >= max_read_failures:
                    raise CameraReadError(
                        f"Camera read failed {consecutive_read_failures} times in a row "
                        f"(camera index {camera_index}, backend {backend}). Possible causes "
                        "include another application using the camera or the camera being "
                        "disconnected; the cause has not been diagnosed. Scanning has stopped."
                    )

                # Replace the last displayed image so no stale result stays on screen.
                status_frame = np.zeros(frame_shape, dtype=np.uint8)
                cv2.putText(
                    status_frame,
                    f"Camera frame unavailable - retrying ({consecutive_read_failures}/{max_read_failures})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2,
                )
                cv2.putText(
                    status_frame,
                    "Scanning paused: no faces are being checked.",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2,
                )
                cv2.imshow(WINDOW_TITLE, status_frame)
                # The pause doubles as the window's event pump; 'q' still quits.
                key = cv2.waitKey(max(1, int(read_retry_delay * 1000))) & 0xFF
                if key == ord("q"):
                    break
                continue

            consecutive_read_failures = 0
            frame_shape = frame.shape if frame.ndim == 3 else frame_shape
            display_frame = frame.copy()
            now_ts = time.time()

            # Dynamic database sync: detect if faces were reset or added from Dashboard
            if now_ts - last_db_check >= 1.0:
                last_db_check = now_ts
                current_mtime = _get_db_mtime()
                if current_mtime != last_db_mtime:
                    last_db_mtime = current_mtime
                    new_enrolled = db.load_enrolled_embeddings()
                    if set(new_enrolled.keys()) != set(enrolled_db.keys()):
                        print(
                            f"[INFO] Enrolled roster synced with DB! Active profiles: {len(new_enrolled)} "
                            f"(was {len(enrolled_db)})"
                        )
                        # Clear in-memory cooldown so newly reset tests aren't locked out
                        last_logged_time.clear()
                    enrolled_db = new_enrolled

            # Heavy work runs only on a timer, never on every frame.
            if now_ts - last_proc >= process_interval:
                last_proc = now_ts

                t0 = time.perf_counter()
                emp_id, sim_score, bbox = recognition_engine.recognize_employee(frame, enrolled_db)
                rec_times.append(time.perf_counter() - t0)

                if bbox is None:
                    status_banner, banner_color = READY_BANNER, GREY

                elif (bbox[2] - bbox[0]) < MIN_FACE_WIDTH:
                    status_banner, banner_color = "Please move closer to the camera", GREY

                else:
                    # Passive liveness runs only on processed frames
                    t1 = time.perf_counter()
                    is_real, liveness_score = liveness_engine.check_passive_liveness(frame, bbox)
                    live_times.append(time.perf_counter() - t1)

                    if not is_real:
                        banner_color = (0, 0, 255)  # Red for spoof attack
                        status_banner = f"SPOOF BLOCKED (Liveness: {liveness_score:.2f})"
                        if now_ts - last_logged_time.get("SPOOF", 0.0) > spoof_cooldown_seconds:
                            db.log_attendance(
                                employee_id=emp_id,
                                spoof_detected=True,
                                liveness_score=liveness_score,
                                similarity_score=sim_score,
                            )
                            last_logged_time["SPOOF"] = now_ts

                    elif emp_id is not None:
                        banner_color = (0, 200, 0)  # Green for verified employee
                        elapsed = now_ts - last_logged_time.get(emp_id, 0.0)
                        if elapsed < lockout_seconds:
                            # Avoid accidental immediate double-logging
                            status_banner = f"Already logged: {emp_id} (Please step aside)"
                        else:
                            action = db.log_attendance(
                                employee_id=emp_id,
                                spoof_detected=False,
                                liveness_score=liveness_score,
                                similarity_score=sim_score,
                            )
                            last_logged_time[emp_id] = now_ts
                            status_banner = f"{action}: {emp_id} (Sim: {sim_score:.2f})"

                    else:
                        banner_color = (0, 165, 255)  # Orange for unknown live person
                        status_banner = "Unrecognized Face (Press 'e' to test-enroll)"

            # Drawing happens every frame using cached results (cheap).
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), banner_color, 2)

            cv2.putText(
                display_frame,
                status_banner,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                banner_color,
                2,
            )
            cv2.imshow(WINDOW_TITLE, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                res = recognition_engine.extract_embedding(frame)
                if res is not None:
                    emb, bbox_enroll = res
                    is_live_enroll, live_score_enroll = liveness_engine.check_passive_liveness(frame, bbox_enroll)
                    if not is_live_enroll:
                        print(f"[SECURITY ALERT] Quick-enroll rejected: Spoof detected (score: {live_score_enroll:.2f})")
                        status_banner, banner_color = f"Enrollment Blocked: Spoof Detected ({live_score_enroll:.2f})", (0, 0, 255)
                    else:
                        enrolled_db["DCCP-KARL-001"] = emb
                        try:
                            db.enroll_employee("DCCP-KARL-001", "Karl Stephen V. Zubiri", emb)
                            print("[INFO] Enrolled DCCP-KARL-001 and persisted to database!")
                            status_banner, banner_color = "Enrolled DCCP-KARL-001 (Saved to DB)!", (0, 200, 0)
                        except Exception as err:
                            print(f"[WARN] In-memory enroll succeeded, DB save failed: {err}")
                            status_banner, banner_color = "Enrolled DCCP-KARL-001 (Session Only)!", (0, 200, 0)
            elif key == ord("q"):
                break
    finally:
        # Each cleanup step runs even if an earlier one fails.
        try:
            cap.release()
        except Exception as err:
            print(f"[WARN] Camera release failed: {err}")
        try:
            cv2.destroyAllWindows()
        except Exception as err:
            print(f"[WARN] Closing scanner windows failed: {err}")
        _print_latency("recognition", rec_times)
        _print_latency("liveness", live_times)


def main() -> int:
    """Runs the scanner and maps known failures to process exit codes."""
    try:
        run_attendance_scanner()
    except CameraReadError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return EXIT_CAMERA_READ_FAILURE
    except ScannerConfigError as err:
        print(f"[ERROR] Invalid scanner configuration: {err}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except KeyboardInterrupt:
        print("[INFO] Scanner interrupted (Ctrl+C). Scanning has stopped.")
        return EXIT_INTERRUPTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
