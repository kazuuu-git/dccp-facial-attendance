import sqlite3
import pandas as pd

def get_db_connection(db_path="attendance.db"):
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn

def fetch_attendance_logs(db_path="attendance.db"):
    """Queries the database for all attendance records."""
    conn = get_db_connection(db_path)
    
    # Selecting the specific data points required by the core architecture
    query = """
        SELECT employee_id, date, time_in, time_out, spoof_status 
        FROM logs 
        ORDER BY date DESC, time_in DESC
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame() # Return empty dataframe on failure
    finally:
        conn.close()

def convert_df_to_csv(df):
    """Converts a Pandas dataframe to a CSV format for Streamlit downloading."""
    return df.to_csv(index=False).encode('utf-8')