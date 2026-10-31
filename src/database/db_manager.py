"""
Milestone 2: SQLite Attendance Database Manager
Owner: Earl (Adapter & Full Implementation)

Provides persistent SQLite storage for:
  1. Enrolled employee facial embeddings (512-D float32 vectors).
  2. Attendance event logs (TIME_IN, TIME_OUT, SPOOF_BLOCKED).

Compatible with:
  - Scanner pipeline (app_scanner.py)
  - Streamlit dashboard (src/dashboard/export_utils.py, ui_views.py)
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Canonical database path: <project_root>/data/attendance.db
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "attendance.db"


class AttendanceDatabase:
    """Manages employee embeddings and attendance logs in SQLite."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            env_path = os.getenv("DCCP_DB_PATH")
            self.db_path = Path(env_path) if env_path else DEFAULT_DB_PATH
        else:
            self.db_path = Path(db_path)

        # Ensure the data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Employees table: stores 512-D float32 embeddings as binary BLOBs
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 2. Attendance logs table: schema matches Earl's Streamlit query
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    date TEXT NOT NULL,
                    time_in TEXT NOT NULL,
                    time_out TEXT NOT NULL,
                    spoof_status TEXT NOT NULL,
                    liveness_score REAL DEFAULT 0.0,
                    similarity_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Performance indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_date ON logs(date);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_employee ON logs(employee_id);"
            )

    def load_enrolled_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Loads all enrolled employee embeddings from the database.

        Returns:
            Dict mapping employee_id to normalized 512-D float32 numpy array.
        """
        enrolled: Dict[str, np.ndarray] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, embedding FROM employees;")
            for row in cursor.fetchall():
                emp_id = row["employee_id"]
                blob = row["embedding"]
                embedding = np.frombuffer(blob, dtype=np.float32).copy()
                if embedding.shape == (512,):
                    enrolled[emp_id] = embedding
                else:
                    print(
                        f"[WARN] Skipping {emp_id}: invalid embedding shape {embedding.shape}"
                    )
        return enrolled

    def enroll_employee(
        self, employee_id: str, full_name: str, embedding: np.ndarray
    ) -> bool:
        """
        Enrolls or updates an employee's 512-D embedding.

        Args:
            employee_id: Unique company ID (e.g. "DCCP-1001").
            full_name: Employee's full name.
            embedding: 512-D numpy vector.

        Returns:
            True if enrollment succeeded.
        """
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        if arr.shape != (512,):
            raise ValueError(f"Expected 512-D embedding, got shape {arr.shape}")

        blob = sqlite3.Binary(arr.tobytes())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO employees (employee_id, full_name, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    embedding = excluded.embedding,
                    created_at = CURRENT_TIMESTAMP;
                """,
                (employee_id.strip(), full_name.strip(), blob),
            )
        print(f"[DB] Successfully enrolled/updated employee {employee_id} ({full_name}).")
        return True

    def get_employee(self, employee_id: str) -> Optional[dict]:
        """Fetches employee record metadata without raw embedding bytes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT employee_id, full_name, created_at FROM employees WHERE employee_id = ?;",
                (employee_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def log_attendance(
        self,
        employee_id: Optional[str],
        spoof_detected: bool,
        liveness_score: float = 0.0,
        similarity_score: float = 0.0,
    ) -> str:
        """
        Records an attendance attempt or spoof alert into the logs table.

        Behavior:
          - If spoof_detected is True: logs a blocked spoof attempt and returns "SPOOF_BLOCKED".
          - If verified employee:
              * If there is an open record for today (time_in present, time_out == '--'),
                updates time_out and returns "TIME_OUT".
              * Otherwise, creates a new record with time_in and returns "TIME_IN".

        Returns:
            "TIME_IN" | "TIME_OUT" | "SPOOF_BLOCKED"
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%I:%M:%S %p")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Case 1: Spoof attempt detected
            if spoof_detected or employee_id is None:
                emp = employee_id or "UNKNOWN"
                status = f"Failed (Spoof Detected, Score: {liveness_score:.2f})"
                cursor.execute(
                    """
                    INSERT INTO logs (employee_id, date, time_in, time_out, spoof_status, liveness_score, similarity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (emp, date_str, "--", "--", status, float(liveness_score), float(similarity_score)),
                )
                print(f"[DB] Logged SPOOF_BLOCKED for {emp} (Score: {liveness_score:.2f})")
                return "SPOOF_BLOCKED"

            # Case 2: Verified employee — find today's latest open session
            cursor.execute(
                """
                SELECT id, time_in, time_out FROM logs
                WHERE employee_id = ? AND date = ? AND time_out = '--'
                ORDER BY id DESC LIMIT 1;
                """,
                (employee_id, date_str),
            )
            open_record = cursor.fetchone()

            if open_record:
                # Close the session with TIME_OUT
                record_id = open_record["id"]
                cursor.execute(
                    """
                    UPDATE logs
                    SET time_out = ?, similarity_score = ?
                    WHERE id = ?;
                    """,
                    (time_str, float(similarity_score), record_id),
                )
                print(f"[DB] Logged TIME_OUT for {employee_id} at {time_str}")
                return "TIME_OUT"
            else:
                # Open a new session with TIME_IN
                cursor.execute(
                    """
                    INSERT INTO logs (employee_id, date, time_in, time_out, spoof_status, liveness_score, similarity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (employee_id, date_str, time_str, "--", "Passed", float(liveness_score), float(similarity_score)),
                )
                print(f"[DB] Logged TIME_IN for {employee_id} at {time_str}")
                return "TIME_IN"

    def reset_database(self, clear_employees: bool = True, clear_logs: bool = True) -> None:
        """
        Clears all records from logs and/or employees tables.
        Useful for resetting test data between test runs.
        """
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                if clear_logs:
                    cursor.execute("DELETE FROM logs;")
                    print("[DB] Cleared all attendance logs.")
                if clear_employees:
                    cursor.execute("DELETE FROM employees;")
                    print("[DB] Cleared all enrolled employee profiles.")
            # VACUUM must run outside an active transaction in autocommit mode
            conn.isolation_level = None
            conn.execute("VACUUM;")
        finally:
            conn.close()
        print("[DB] Database reset complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCCP Attendance Database Utility")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset entire database (clears both enrolled employees and attendance logs)",
    )
    parser.add_argument(
        "--clear-logs",
        action="store_true",
        help="Clear only attendance logs (preserves enrolled employee profiles)",
    )
    parser.add_argument(
        "--clear-employees",
        action="store_true",
        help="Clear only enrolled employee embeddings",
    )
    args = parser.parse_args()

    db = AttendanceDatabase()

    if args.reset:
        db.reset_database(clear_employees=True, clear_logs=True)
    elif args.clear_logs:
        db.reset_database(clear_employees=False, clear_logs=True)
    elif args.clear_employees:
        db.reset_database(clear_employees=True, clear_logs=False)
    else:
        enrolled = db.load_enrolled_embeddings()
        print(f"[INFO] Database at: {db.db_path}")
        print(f"[INFO] Total enrolled employee profile(s): {len(enrolled)}")
