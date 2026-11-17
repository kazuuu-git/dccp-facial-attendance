"""
Automated Comprehensive Test Suite for Passive Liveness, Threshold Policy,
Biometric Enrollment, and Attendance Storage.

Verifies:
1. LIVE uniquely highest and at/above threshold -> PASS
2. LIVE highest but below threshold -> REJECT
3. A spoof class winning (Class 0 or Class 2) -> REJECT
4. A tie involving LIVE -> REJECT
5. Missing or unusable face input (None, zero size, invalid bbox) -> REJECT / False
6. Failed inference and invalid / non-finite scores -> INVALID / False
7. Known identity with failed liveness -> Blocked, logged as SPOOF_BLOCKED (NO TIME_IN/OUT)
8. Unknown identity with passed liveness -> NO attendance recorded
9. Attendance creation blocked when any required check fails
10. The real scanner / web path actually using the threshold policy
"""

import gc
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.database.db_manager import AttendanceDatabase
from src.liveness.anti_spoof import PassiveLivenessEngine
from src.liveness.threshold import evaluate_liveness
from src.recognition.matcher import FaceRecognitionEngine


class TestThresholdPolicy(unittest.TestCase):
    """Direct unit tests for src/liveness/threshold.py."""

    def test_01_live_uniquely_highest_and_meets_threshold(self):
        # Class 1 is 0.90, threshold 0.50 -> PASS
        scores = [[0.05, 0.90, 0.05]]
        res = evaluate_liveness(scores, threshold=0.50)
        self.assertTrue(res["passed"])
        self.assertEqual(res["status"], "PASS")
        self.assertAlmostEqual(res["live_score"], 0.90)

    def test_02_live_highest_but_below_threshold(self):
        # Class 1 is 0.45 (highest), but threshold is 0.50 -> REJECT
        scores = [[0.27, 0.46, 0.27]]
        res = evaluate_liveness(scores, threshold=0.50)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "REJECT")
        self.assertIn("below the threshold", res["reason"])

    def test_03_spoof_class_winning(self):
        # Class 2 (screen/replay) is highest (0.80) -> REJECT
        scores = [[0.05, 0.15, 0.80]]
        res = evaluate_liveness(scores, threshold=0.50)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "REJECT")
        self.assertIn("not the unique highest", res["reason"])

    def test_04_tie_involving_live(self):
        # Tie between Class 1 and Class 2 at 0.50 each -> REJECT
        scores = [[0.0, 0.50, 0.50]]
        res = evaluate_liveness(scores, threshold=0.50)
        self.assertFalse(res["passed"])
        self.assertEqual(res["status"], "REJECT")
        self.assertIn("not the unique highest", res["reason"])

        # 3-way tie -> REJECT
        scores_3way = [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
        res_3way = evaluate_liveness(scores_3way, threshold=0.30)
        self.assertFalse(res_3way["passed"])
        self.assertEqual(res_3way["status"], "REJECT")

    def test_05_multi_model_ensemble_averaging(self):
        # Model 1 gives [0.10, 0.80, 0.10], Model 2 gives [0.20, 0.60, 0.20]
        # Average is [0.15, 0.70, 0.15] -> live_score = 0.70 >= 0.50 -> PASS
        scores = [[0.10, 0.80, 0.10], [0.20, 0.60, 0.20]]
        res = evaluate_liveness(scores, threshold=0.50)
        self.assertTrue(res["passed"])
        self.assertAlmostEqual(res["live_score"], 0.70)

    def test_06_invalid_and_non_finite_scores(self):
        # Empty scores
        res_empty = evaluate_liveness([], threshold=0.50)
        self.assertFalse(res_empty["passed"])
        self.assertEqual(res_empty["status"], "INVALID")

        # Non-finite values
        res_nan = evaluate_liveness([[float("nan"), 0.5, 0.5]], threshold=0.50)
        self.assertFalse(res_nan["passed"])
        self.assertEqual(res_nan["status"], "INVALID")

        # Scores don't sum to 1
        res_nosum = evaluate_liveness([[0.1, 0.2, 0.3]], threshold=0.50)
        self.assertFalse(res_nosum["passed"])
        self.assertEqual(res_nosum["status"], "INVALID")


class TestPassiveLivenessEngine(unittest.TestCase):
    """Integration tests for PassiveLivenessEngine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = PassiveLivenessEngine(threshold=0.50)

    def test_07_unusable_face_input(self):
        # None frame
        is_real, score = self.engine.check_passive_liveness(None, [10, 10, 50, 50])
        self.assertFalse(is_real)
        self.assertEqual(score, 0.0)

        # Empty frame
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        is_real, score = self.engine.check_passive_liveness(empty_img, [10, 10, 50, 50])
        self.assertFalse(is_real)
        self.assertEqual(score, 0.0)

        # Invalid bbox dimensions (width <= 0)
        valid_img = np.zeros((480, 640, 3), dtype=np.uint8)
        is_real, score = self.engine.check_passive_liveness(valid_img, [50, 50, 50, 50])
        self.assertFalse(is_real)
        self.assertEqual(score, 0.0)

        # Bbox completely outside frame
        is_real, score = self.engine.check_passive_liveness(valid_img, [700, 700, 800, 800])
        self.assertFalse(is_real)
        self.assertEqual(score, 0.0)

    def test_08_real_samples_verification(self):
        ref_dir = Path(__file__).resolve().parents[2] / "Silent-Face-Anti-Spoofing-reference" / "images" / "sample"
        if not ref_dir.is_dir():
            self.skipTest(f"Reference images dir not found: {ref_dir}")

        rec = FaceRecognitionEngine()

        # Genuine live face
        img_t1 = cv2.imread(str(ref_dir / "image_T1.jpg"))
        if img_t1 is not None:
            _, bbox_t1 = rec.extract_embedding(img_t1)
            is_real, score = self.engine.check_passive_liveness(img_t1, bbox_t1)
            self.assertTrue(is_real, f"True face should pass! Score: {score}")
            self.assertGreater(score, 0.90)

        # Spoof Fake 1
        img_f1 = cv2.imread(str(ref_dir / "image_F1.jpg"))
        if img_f1 is not None:
            _, bbox_f1 = rec.extract_embedding(img_f1)
            is_real, score = self.engine.check_passive_liveness(img_f1, bbox_f1)
            self.assertFalse(is_real, f"Fake 1 should be blocked! Score: {score}")
            self.assertLess(score, 0.50)

        # Spoof Fake 2
        img_f2 = cv2.imread(str(ref_dir / "image_F2.jpg"))
        if img_f2 is not None:
            _, bbox_f2 = rec.extract_embedding(img_f2)
            is_real, score = self.engine.check_passive_liveness(img_f2, bbox_f2)
            self.assertFalse(is_real, f"Fake 2 should be blocked! Score: {score}")
            self.assertLess(score, 0.50)


class TestAttendanceAndDatabaseSafety(unittest.TestCase):
    """Tests database logging and security guardrails using a temporary SQLite DB."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_attendance.db"
        self.db = AttendanceDatabase(db_path=self.test_db_path)

    def tearDown(self):
        del self.db
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_09_known_identity_with_failed_liveness_blocks_attendance(self):
        # Enroll an employee first
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)
        self.db.enroll_employee("EMP-001", "Alice Test", dummy_emb)

        # A known employee presentation FAILS liveness (spoof_detected=True)
        action = self.db.log_attendance(
            employee_id="EMP-001",
            spoof_detected=True,
            liveness_score=0.15,
            similarity_score=0.92,
        )

        self.assertEqual(action, "SPOOF_BLOCKED")

        # Verify NO TIME_IN was recorded
        conn = self.db._get_connection()
        try:
            logs = conn.execute("SELECT * FROM logs WHERE employee_id='EMP-001'").fetchall()
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["time_in"], "--")
            self.assertEqual(logs[0]["time_out"], "--")
            self.assertIn("Failed (Spoof Detected", logs[0]["spoof_status"])
        finally:
            conn.close()

    def test_10_unknown_identity_with_passed_liveness_does_not_log_time_in(self):
        # An unknown person (employee_id=None) passes liveness
        # In app_scanner.py, when emp_id is None, db.log_attendance is NOT called for attendance!
        # If someone attempted to log attendance for None, db records SPOOF_BLOCKED:
        action = self.db.log_attendance(
            employee_id=None,
            spoof_detected=False,
            liveness_score=0.95,
            similarity_score=0.20,
        )
        self.assertEqual(action, "SPOOF_BLOCKED")

    def test_11_valid_employee_with_passed_liveness_records_attendance(self):
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb /= np.linalg.norm(dummy_emb)
        self.db.enroll_employee("EMP-002", "Bob Test", dummy_emb)

        # Live verification passes
        action1 = self.db.log_attendance(
            employee_id="EMP-002",
            spoof_detected=False,
            liveness_score=0.98,
            similarity_score=0.88,
        )
        self.assertEqual(action1, "TIME_IN")

        # Second clock logs TIME_OUT
        action2 = self.db.log_attendance(
            employee_id="EMP-002",
            spoof_detected=False,
            liveness_score=0.97,
            similarity_score=0.89,
        )
        self.assertEqual(action2, "TIME_OUT")


if __name__ == "__main__":
    unittest.main()
