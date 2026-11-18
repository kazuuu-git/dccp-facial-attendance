"""
Read-only access to enrolled employee embeddings for scanner diagnostics.

Guarantees (by design, see tests/test_readonly_roster.py):
  - Never creates the database file or its parent folder.
  - Opens SQLite only with a URI "mode=ro" connection; there is no writable
    fallback, no journal-mode change and no immutable=1.
  - Databases in WAL journal mode are refused (reported as unavailable), because
    a read-only WAL reader may need to create -shm/-wal side files.
  - Has no write methods at all.

Roster states:
  AVAILABLE    last refresh succeeded (the roster may legitimately be empty)
  CACHED       last refresh failed transiently; identities from the previous
               successful refresh are retained and must be labelled as cached
  UNAVAILABLE  no usable roster (missing/invalid database or no successful read)
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from src.database.db_manager import DEFAULT_DB_PATH

AVAILABLE = "available"
CACHED = "cached"
UNAVAILABLE = "unavailable"

EMBEDDING_DIM = 512
SQLITE_HEADER = b"SQLite format 3\x00"
READ_TIMEOUT_SECONDS = 0.5


class ReadOnlyEmployeeRoster:
    """Loads enrolled embeddings through a read-only SQLite connection."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        if db_path is None:
            env_path = os.getenv("DCCP_DB_PATH")
            self.db_path = Path(env_path) if env_path else DEFAULT_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.state = UNAVAILABLE
        self.reason: Optional[str] = "not loaded yet"
        self.skipped_invalid = 0
        self._embeddings: Dict[str, np.ndarray] = {}
        self.refresh()

    @property
    def embeddings(self) -> Dict[str, np.ndarray]:
        return dict(self._embeddings)

    @property
    def employee_count(self) -> int:
        return len(self._embeddings)

    def load_enrolled_embeddings(self) -> Dict[str, np.ndarray]:
        """Refreshes, then returns the current roster (possibly cached or empty)."""
        self.refresh()
        return self.embeddings

    def refresh(self) -> None:
        """Re-reads the roster. Never writes; never raises for database problems."""
        definitive_problem = self._precheck_file()
        if definitive_problem is not None:
            self._set_unavailable(definitive_problem)
            return

        try:
            conn = self._connect_readonly()
        except sqlite3.Error as err:
            self._read_failed(f"read error: {err}")
            return

        try:
            rows = conn.execute("SELECT employee_id, embedding FROM employees;").fetchall()
        except sqlite3.OperationalError as err:
            if "no such table" in str(err).lower():
                self._set_unavailable("database has no employees table")
            else:
                self._read_failed(f"read error: {err}")
            return
        except sqlite3.Error as err:
            self._read_failed(f"read error: {err}")
            return
        finally:
            conn.close()

        embeddings: Dict[str, np.ndarray] = {}
        skipped = 0
        for emp_id, blob in rows:
            try:
                embedding = np.frombuffer(blob, dtype=np.float32).copy()
            except (TypeError, ValueError):
                skipped += 1
                continue
            if embedding.shape == (EMBEDDING_DIM,):
                embeddings[emp_id] = embedding
            else:
                skipped += 1

        # A successful read replaces the roster, even when it is empty.
        self._embeddings = embeddings
        self.skipped_invalid = skipped
        self.state = AVAILABLE
        self.reason = None

    def _connect_readonly(self) -> sqlite3.Connection:
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=READ_TIMEOUT_SECONDS)

    def _precheck_file(self) -> Optional[str]:
        """Checks the file without SQLite. Returns a reason if it must not be opened."""
        if not self.db_path.is_file():
            return "database file not found"
        try:
            with open(self.db_path, "rb") as fh:
                header = fh.read(100)
        except OSError as err:
            return f"database file cannot be read ({err.strerror or err})"
        if len(header) == 0:
            return "database file is empty"
        if len(header) < 100 or not header.startswith(SQLITE_HEADER):
            return "file is not a SQLite database"
        # Header bytes 18/19 are the write/read format versions; 2 means WAL mode.
        if header[18] == 2 or header[19] == 2:
            return (
                "database uses WAL journal mode; read-only access could create side "
                "files, so identity matching is disabled"
            )
        return None

    def _set_unavailable(self, reason: str) -> None:
        self._embeddings = {}
        self.skipped_invalid = 0
        self.state = UNAVAILABLE
        self.reason = reason

    def _read_failed(self, reason: str) -> None:
        """Transient failure: keep a previous successful roster, labelled as cached."""
        if self.state in (AVAILABLE, CACHED):
            self.state = CACHED
            self.reason = reason
        else:
            self._set_unavailable(reason)
