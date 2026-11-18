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
    from src.liveness.anti_spoof import PassiveLivenessEngine, decision_to_result
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

# Diagnostics mode reads enrolled identities through a read-only connection only.
from src.database import readonly_roster
from src.database.readonly_roster import ReadOnlyEmployeeRoster

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

BLOCKED_BANNER = "BLOCKED: liveness check failed"
DIAG_WINDOW_TITLE = "DCCP Scanner - DIAGNOSTICS (nothing recorded)"
DIAG_HEADER = "DIAGNOSTICS MODE - no attendance or enrollment is recorded"
DIAG_FOOTER = "Model scores are not calibrated probabilities."
DIAG_NO_FACE = "No face detected"
DIAG_ENROLL_DISABLED = "Enrollment disabled in diagnostics mode"


def format_liveness_line(decision: dict) -> Tuple[str, Tuple[int, int, int]]:
    """Overlay text for a liveness decision. Never shows percentages or probabilities.

    INVALID shows no score, because the check could not be completed.
    """
    status = decision.get("status")
    score = decision.get("live_score")
    threshold = decision.get("threshold")
    threshold_text = f" | threshold {threshold:.2f}" if threshold is not None else ""
    if status in ("PASS", "REJECT") and score is not None:
        color = GREEN if status == "PASS" else RED
        return f"Liveness: {status} | model live score {score:.2f}{threshold_text}", color
    return f"Liveness: INVALID - check could not be completed{threshold_text}", ORANGE


def format_identity_line(roster, emp_id: Optional[str], sim_score: float) -> str:
    """Diagnostics identity text. A read problem is never shown as an unknown person."""
    if roster.state == readonly_roster.UNAVAILABLE:
        return f"Identity matching off: {roster.reason}"
    cached = " [cached roster: latest refresh failed]" if roster.state == readonly_roster.CACHED else ""
    if roster.employee_count == 0:
        return f"Identity: no employees enrolled{cached}"
    if emp_id is not None:
        return f"Identity: {emp_id} (similarity {sim_score:.2f}){cached}"
    return f"Identity: no enrolled match (best similarity {sim_score:.2f}){cached}"


def liveness_result_is_fresh(result_ts: Optional[float], now_ts: float, process_interval: float) -> bool:
    """Liveness/identity results older than max(1s, 3 x interval) are not displayed."""
    if result_ts is None:
        return False
    return (now_ts - result_ts) <= max(1.0, 3.0 * process_interval)


def draw_label(img: np.ndarray, text: str, y: int, color, scale: float = 0.6, thickness: int = 1) -> int:
    """Draws text on a dark backdrop, shrunk if needed to stay inside the frame.

    Returns the baseline y for the next line.
    """
    h, w = img.shape[:2]
    margin = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    max_w = max(1, w - 2 * margin)
    if tw > max_w:
        scale = max(0.3, scale * max_w / tw)
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    y = int(min(max(y, th + margin), h - base - margin))
    cv2.rectangle(img, (margin - 4, y - th - 4), (min(w - 1, margin + tw + 4), y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (margin, y), font, scale, color, thickness, cv2.LINE_AA)
    return y + th + base + 12


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
    diagnostics: bool = False,
) -> None:
    """Runs the real-time DCCP employee attendance kiosk loop.

    With diagnostics=True the same detection, recognition and liveness pipeline
    runs, but nothing is written: AttendanceDatabase is never created, enrolled
    identities are read through a read-only connection, and quick-enroll is
    disabled.

    Raises CameraReadError if the camera returns `max_read_failures` unusable
    frames in a row, and ScannerConfigError for invalid configuration.
    """
    # Validate configuration before loading models or touching the camera.
    if max_read_failures is None:
        max_read_failures = parse_max_read_failures(os.getenv("DCCP_MAX_READ_FAILURES"))
    elif isinstance(max_read_failures, bool) or not isinstance(max_read_failures, int) or max_read_failures < 1:
        raise ScannerConfigError(f"max_read_failures must be a positive integer, got {max_read_failures!r}.")

    db = None
    roster = None
    if diagnostics:
        # Never AttendanceDatabase here: its constructor creates folders and tables.
        roster = ReadOnlyEmployeeRoster()
    else:
        db = AttendanceDatabase()
    liveness_engine = PassiveLivenessEngine()
    recognition_engine = FaceRecognitionEngine()

    if diagnostics:
        enrolled_db = roster.embeddings
        print("[DIAGNOSTICS] No attendance records, blocked attempts or enrollments will be written.")
        if roster.state == readonly_roster.UNAVAILABLE:
            print(f"[DIAGNOSTICS] Identity matching off ({roster.reason}); running liveness-only.")
        else:
            print(f"[DIAGNOSTICS] Read-only roster loaded: {roster.employee_count} enrolled profile(s).")
    else:
        enrolled_db = db.load_enrolled_embeddings()
        print(f"[INFO] Loaded {len(enrolled_db)} enrolled DCCP employee profile(s).")
    print(f"[INFO] Recognition interval: {process_interval:.2f}s | Min face width: {MIN_FACE_WIDTH}px | Lockout: {lockout_seconds:.0f}s")
    if diagnostics:
        print("[CONTROLS] Press 'q' to quit. Enrollment is disabled in diagnostics mode.")
    else:
        print("[CONTROLS] Press 'e' to quick-enroll your face for testing | Press 'q' to quit.")

    window_title = DIAG_WINDOW_TITLE if diagnostics else WINDOW_TITLE
    ready_banner = DIAG_NO_FACE if diagnostics else READY_BANNER
    watched_db_path = roster.db_path if diagnostics else db.db_path

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
        status_banner = ready_banner
        banner_color = GREY

        bbox: Optional[list] = None  # cached between recognition cycles (for drawing only)
        last_proc = 0.0
        consecutive_read_failures = 0
        frame_shape = (480, 640, 3)

        # Display-only results of the latest processing cycle.
        liveness_decision: Optional[dict] = None
        identity_text: Optional[str] = None
        result_ts: Optional[float] = None
        diag_notice_until = 0.0  # diagnostics-only "enrollment disabled" notice

        def _get_db_mtime() -> float:
            try:
                return os.path.getmtime(watched_db_path) if os.path.exists(watched_db_path) else 0.0
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
                status_banner, banner_color = ready_banner, GREY
                last_proc = 0.0
                liveness_decision, identity_text, result_ts = None, None, None

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
                cv2.imshow(window_title, status_frame)
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
                if diagnostics:
                    # Read-only refresh; retried while the roster is not AVAILABLE.
                    if current_mtime != last_db_mtime or roster.state != readonly_roster.AVAILABLE:
                        last_db_mtime = current_mtime
                        roster.refresh()
                        enrolled_db = roster.embeddings
                elif current_mtime != last_db_mtime:
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
                # Results from an earlier cycle never carry over to this one.
                liveness_decision, identity_text, result_ts = None, None, None

                t0 = time.perf_counter()
                emp_id, sim_score, bbox = recognition_engine.recognize_employee(frame, enrolled_db)
                rec_times.append(time.perf_counter() - t0)

                if bbox is None:
                    status_banner, banner_color = ready_banner, GREY

                elif (bbox[2] - bbox[0]) < MIN_FACE_WIDTH:
                    status_banner, banner_color = "Please move closer to the camera", GREY

                else:
                    # Passive liveness runs only on processed frames, once: the same
                    # decision drives attendance and the overlay.
                    t1 = time.perf_counter()
                    decision = liveness_engine.evaluate(frame, bbox)
                    live_times.append(time.perf_counter() - t1)
                    is_real, liveness_score = decision_to_result(decision)
                    liveness_decision, result_ts = decision, now_ts

                    if diagnostics:
                        # Display only: no attendance, blocked-attempt or enrollment writes.
                        identity_text = format_identity_line(roster, emp_id, sim_score)
                        status_banner, banner_color = "Face detected", GREY

                    elif not is_real:
                        banner_color = RED
                        status_banner = BLOCKED_BANNER
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

            show_result = liveness_decision is not None and liveness_result_is_fresh(
                result_ts, now_ts, process_interval
            )
            if diagnostics:
                y = draw_label(display_frame, DIAG_HEADER, 28, YELLOW, 0.6, 2)
                y = draw_label(display_frame, status_banner, y, banner_color)
                if now_ts < diag_notice_until:
                    y = draw_label(display_frame, DIAG_ENROLL_DISABLED, y, ORANGE)
                if show_result:
                    if identity_text:
                        y = draw_label(display_frame, identity_text, y, GREY)
                    live_text, live_color = format_liveness_line(liveness_decision)
                    draw_label(display_frame, live_text, y, live_color)
                draw_label(display_frame, DIAG_FOOTER, display_frame.shape[0] - 12, GREY, 0.5)
            else:
                y = draw_label(display_frame, status_banner, 40, banner_color, 0.75, 2)
                if show_result:
                    live_text, live_color = format_liveness_line(liveness_decision)
                    draw_label(display_frame, live_text, y, live_color)
            cv2.imshow(window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e") and diagnostics:
                # No extraction, liveness check or database access for enrollment.
                diag_notice_until = time.time() + 2.0
            elif key == ord("e"):
                res = recognition_engine.extract_embedding(frame)
                if res is not None:
                    emb, bbox_enroll = res
                    is_live_enroll, live_score_enroll = liveness_engine.check_passive_liveness(frame, bbox_enroll)
                    if not is_live_enroll:
                        print("[SECURITY ALERT] Quick-enroll rejected: liveness check failed.")
                        status_banner, banner_color = f"Enrollment {BLOCKED_BANNER}", RED
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


def main(argv: Optional[List[str]] = None) -> int:
    """Runs the scanner and maps known failures to process exit codes."""
    import argparse

    parser = argparse.ArgumentParser(description="DCCP attendance scanner")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show liveness/identity results only; write no attendance, blocked attempts or enrollments.",
    )
    args = parser.parse_args(argv if argv is not None else [])
    try:
        run_attendance_scanner(diagnostics=args.diagnostics)
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
    sys.exit(main(sys.argv[1:]))
