import os
import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "attendance.db"


def get_db_connection(db_path=None):
    """Establishes a connection to the SQLite database."""
    if db_path is None:
        env_path = os.getenv("DCCP_DB_PATH")
        path = Path(env_path) if env_path else DEFAULT_DB_PATH
    else:
        path = Path(db_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    return conn


def fetch_attendance_logs(db_path=None):
    """Queries the database for all attendance records."""
    conn = get_db_connection(db_path)

    # Selecting the specific data points required by the core architecture
    query = """
        SELECT employee_id, date, time_in, time_out, spoof_status 
        FROM logs 
        ORDER BY id DESC, date DESC
    """

    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame()  # Return empty dataframe on failure
    finally:
        conn.close()


def convert_df_to_csv(df):
    """Converts a Pandas dataframe to a CSV format for Streamlit downloading."""
    return df.to_csv(index=False).encode("utf-8")
