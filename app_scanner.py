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
"""

import os
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

# Import Corliss's Passive Liveness module
try:
    from src.liveness.anti_spoof import PassiveLivenessEngine
    print("[INFO] Loaded REAL PassiveLivenessEngine.")
except Exception as e:
    print(f"[WARN] Failed to load real PassiveLivenessEngine ({e}). Using temporary mock.")

    class PassiveLivenessEngine:  # type: ignore[no-redef]
        """Temporary fallback if liveness dependencies are not installed."""

        def check_passive_liveness(
            self, frame_bgr: np.ndarray, bbox: Optional[list] = None
        ) -> Tuple[bool, float]:
            return True, 0.99


# Import Earl's SQLite Database module
try:
    from src.database.db_manager import AttendanceDatabase
    print("[INFO] Loaded REAL AttendanceDatabase.")
except Exception as e:
    print(f"[WARN] Failed to load real AttendanceDatabase ({e}). Using temporary mock.")

    class AttendanceDatabase:  # type: ignore[no-redef]
        """Temporary in-memory fallback if SQLite module fails."""

        def __init__(self) -> None:
            self._enrolled: Dict[str, np.ndarray] = {}
            self._toggle_state: Dict[str, str] = {}

        def load_enrolled_embeddings(self) -> Dict[str, np.ndarray]:
            return self._enrolled

        def enroll_employee(
            self, employee_id: str, full_name: str, embedding: np.ndarray
        ) -> bool:
            self._enrolled[employee_id] = embedding
            return True

        def log_attendance(
            self,
            employee_id: Optional[str],
            spoof_detected: bool,
            liveness_score: float = 0.0,
            similarity_score: float = 0.0,
        ) -> str:
            if spoof_detected or employee_id is None:
                print(f"[MOCK DB] Logged SPOOF_BLOCKED (Liveness={liveness_score:.2f})")
                return "SPOOF_BLOCKED"
            prev = self._toggle_state.get(employee_id, "TIME_OUT")
            action = "TIME_IN" if prev == "TIME_OUT" else "TIME_OUT"
            self._toggle_state[employee_id] = action
            print(f"[MOCK DB] Logged {action} for {employee_id} (Sim={similarity_score:.2f})")
            return action


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
) -> None:
    """Runs the real-time DCCP employee attendance kiosk loop."""
    db = AttendanceDatabase()
    liveness_engine = PassiveLivenessEngine()
    recognition_engine = FaceRecognitionEngine()

    enrolled_db = db.load_enrolled_embeddings()
    print(f"[INFO] Loaded {len(enrolled_db)} enrolled DCCP employee profile(s).")
    print(f"[INFO] Recognition interval: {process_interval:.2f}s | Min face width: {MIN_FACE_WIDTH}px | Lockout: {lockout_seconds:.0f}s")
    print("[CONTROLS] Press 'e' to quick-enroll your face for testing | Press 'q' to quit.")

    # Open the camera ONCE, then configure it.
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    last_logged_time: Dict[str, float] = {}
    status_banner = READY_BANNER
    banner_color = GREY

    bbox: Optional[list] = None  # cached between recognition cycles (for drawing only)
    last_proc = 0.0
    rec_times: List[float] = []
    live_times: List[float] = []

    def _get_db_mtime() -> float:
        try:
            return os.path.getmtime(db.db_path) if os.path.exists(db.db_path) else 0.0
        except OSError:
            return 0.0

    last_db_mtime = _get_db_mtime()
    last_db_check = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
            cv2.imshow("DCCP AI Facial Attendance Scanner", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                res = recognition_engine.extract_embedding(frame)
                if res is not None:
                    emb, _ = res
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
        cap.release()
        cv2.destroyAllWindows()
        _print_latency("recognition", rec_times)
        _print_latency("liveness", live_times)


if __name__ == "__main__":
    run_attendance_scanner()
