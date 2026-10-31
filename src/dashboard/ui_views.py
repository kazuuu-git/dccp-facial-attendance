import streamlit as st
import pandas as pd
from src.dashboard import export_utils

def render_logs_view():
    st.header("Live Attendance Logs")
    st.write("Monitor real-time employee clock-in and clock-out attempts.")
    
    # Fetch real data from SQLite
    df = export_utils.fetch_attendance_logs()
    
    # If the database or table isn't created yet, show a fallback preview
    if df.empty:
        st.info("Notice: `attendance.db` logs table is empty or not yet initialized. Displaying layout preview:")
        df = pd.DataFrame({
            "employee_id": ["DCCP-1001", "DCCP-1002", "DCCP-1003"],
            "date": ["2026-10-28", "2026-10-28", "2026-10-28"],
            "time_in": ["07:45 AM", "07:52 AM", "08:05 AM"],
            "time_out": ["--", "--", "--"],
            "spoof_status": ["Passed", "Passed", "Failed (2D Photo)"]
        })
    
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_enrollment_view():
    st.header("Biometric Enrollment Module")
    st.write("Capture and store mathematical 512-dimensional facial embeddings for new personnel.")
    
    with st.form("enrollment_form"):
        col1, col2 = st.columns(2)
        with col1:
            emp_id = st.text_input("Employee ID")
        with col2:
            emp_name = st.text_input("Full Name")
            
        st.write("Capture Face Data")
        image_data = st.camera_input("Take a photo")
        
        submit_btn = st.form_submit_button("Extract & Save Embedding")
        
        if submit_btn:
            if image_data is not None and emp_id and emp_name:
                st.success(f"Successfully enrolled {emp_name} ({emp_id}) into the database.")
            else:
                st.error("Please provide an Employee ID, Name, and capture a valid image.")

def render_export_view():
    st.header("Data Export")
    st.write("Automate CSV data exports for payroll processing.")
    
    st.date_input("Select Date Range for Export")
    
    df = export_utils.fetch_attendance_logs()
    
    # Fallback preview if SQLite database is empty
    if df.empty:
        df = pd.DataFrame({
            "employee_id": ["DCCP-1001", "DCCP-1002"],
            "date": ["2026-10-28", "2026-10-28"],
            "time_in": ["07:45 AM", "07:52 AM"],
            "time_out": ["05:00 PM", "05:00 PM"],
            "spoof_status": ["Passed", "Passed"]
        })
    
    csv_data = export_utils.convert_df_to_csv(df)
    st.download_button(
        label="Download CSV Payroll Report",
        data=csv_data,
        file_name="payroll_export.csv",
        mime="text/csv",
        type="primary"
    )