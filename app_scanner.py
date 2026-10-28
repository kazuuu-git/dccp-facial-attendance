"""
Milestone 3: Live Camera Attendance Scanner Pipeline
Owner: Karl Stephen V. Zubiri

Includes safe fallback mocks so it can run and test app_scanner.py even before
Corliss (src/liveness/anti_spoof.py) and Earl (src/database/db_manager.py) push their files.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from src.recognition.matcher import FaceRecognitionEngine

# Safe import for Corliss's Passive Liveness module (falls back to mock until merged)
try:
    from src.liveness.anti_spoof import PassiveLivenessEngine
except (ImportError, ModuleNotFoundError):

    class PassiveLivenessEngine:  # type: ignore[no-redef]
        """Temporary fallback until Corliss pushes src/liveness/anti_spoof.py."""

        def check_passive_liveness(
            self, frame_bgr: np.ndarray, bbox: Optional[list] = None
        ) -> Tuple[bool, float]:
            return True, 0.99


# Safe import for Earl's SQLite Database module (falls back to in-memory mock until merged)
try:
    from src.database.db_manager import AttendanceDatabase
except (ImportError, ModuleNotFoundError):

    class AttendanceDatabase:  # type: ignore[no-redef]
        """Temporary in-memory fallback until Earl pushes src/database/db_manager.py."""

        def __init__(self) -> None:
            self._enrolled: Dict[str, np.ndarray] = {}
            self._toggle_state: Dict[str, str] = {}

        def load_enrolled_embeddings(self) -> Dict[str, np.ndarray]:
            return self._enrolled

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


def run_attendance_scanner(camera_index: int = 0, cooldown_seconds: float = 4.0) -> None:
    """Runs the real-time DCCP employee attendance kiosk loop."""
    db = AttendanceDatabase()
    liveness_engine = PassiveLivenessEngine()
    recognition_engine = FaceRecognitionEngine()

    enrolled_db = db.load_enrolled_embeddings()
    print(f"[INFO] Loaded {len(enrolled_db)} enrolled DCCP employee profile(s).")
    print("[CONTROLS] Press 'e' to quick-enroll your face for testing | Press 'q' to quit.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    last_logged_time: dict[str, float] = {}
    status_banner = "Ready: Look at camera (Press 'e' to test-enroll)"
    banner_color = (200, 200, 200)

    PROCESS_EVERY_N = 5
    frame_idx = 0
    emp_id, sim_score, bbox = None, 0.0, None
    is_real, liveness_score = True, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_frame = frame.copy()

        if frame_idx % PROCESS_EVERY_N == 0:
            emp_id, sim_score, bbox = recognition_engine.recognize_employee(frame, enrolled_db)
            if bbox is not None:
                is_real, liveness_score = liveness_engine.check_passive_liveness(frame, bbox)
            else:
                is_real, liveness_score = True, 0.0
        frame_idx += 1

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            now_ts = time.time()
            cooldown_key = emp_id if emp_id is not None else "SPOOF"

            if not is_real:
                banner_color = (0, 0, 255)  # Red for spoof attack
                status_banner = f"SPOOF BLOCKED (Liveness: {liveness_score:.2f})"
                if now_ts - last_logged_time.get("SPOOF", 0.0) > cooldown_seconds:
                    db.log_attendance(
                        employee_id=emp_id,
                        spoof_detected=True,
                        liveness_score=liveness_score,
                        similarity_score=sim_score,
                    )
                    last_logged_time["SPOOF"] = now_ts

            elif emp_id is not None:
                banner_color = (0, 200, 0)  # Green for verified employee
                if now_ts - last_logged_time.get(cooldown_key, 0.0) > cooldown_seconds:
                    action = db.log_attendance(
                        employee_id=emp_id,
                        spoof_detected=False,
                        liveness_score=liveness_score,
                        similarity_score=sim_score,
                    )
                    last_logged_time[cooldown_key] = now_ts
                    status_banner = f"{action}: {emp_id} (Sim: {sim_score:.2f})"
            else:
                banner_color = (0, 165, 255)  # Orange for unknown live person
                status_banner = "Unrecognized Face (Press 'e' to test-enroll)"

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
                print("[INFO] Enrolled DCCP-KARL-001 into active session memory!")
                status_banner = "Enrolled DCCP-KARL-001!"
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance_scanner()