import streamlit as st
from src.dashboard import ui_views

st.set_page_config(
    page_title="HR Admin Dashboard | DCCP",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("HR Admin Navigation")
    st.sidebar.write("DCCP Attendance Logging System")
    
    menu_options = {
        "Live Attendance Logs": ui_views.render_logs_view,
        "Enroll New Employee": ui_views.render_enrollment_view,
        "Export Payroll Data": ui_views.render_export_view
    }
    
    selection = st.sidebar.radio("Go to:", list(menu_options.keys()))
    
    # Execute selected view
    menu_options[selection]()

if __name__ == "__main__":
    main()