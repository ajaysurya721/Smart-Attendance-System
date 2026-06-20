import streamlit as st
import cv2
import pandas as pd
import os
import numpy as np
import time
from datetime import datetime

# Safely import WebRTC components to prevent crash if running strictly locally
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Set page configuration for a spacious, modern corporate system layout
st.set_page_config(page_title="AI Smart Attendance", page_icon="🛡️", layout="wide")

# Custom UI Styling to fill whitespace beautifully and add modern dark-blue design accents
st.markdown("""
    <style>
    .main-title { font-size:40px !important; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size:16px !important; color: #4B5563; margin-bottom: 25px; }
    .metric-box { background-color: #F8FAFC; padding: 22px; border-radius: 12px; border: 1px solid #E2E8F0; border-top: 4px solid #3B82F6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .status-badge { background-color: #ECFDF5; color: #065F46; padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 13px; border: 1px solid #A7F3D0; }
    </style>
""", unsafe_allow_html=True)

# Top Banner Brand Block
st.markdown('<p class="main-title">🛡️ AI Enterprise Smart Attendance Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated real-time cloud facial analytics, logging pipeline, and workforce insights dashboard.</p>', unsafe_allow_html=True)

# Initialize standard OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- OPTIMIZED PIXEL MATCHING ENGINE ---
def recognize_face_pure_math(live_face_gray, dataset_path="dataset"):
    if not os.path.exists(dataset_path):
        return "UNKNOWN", "Unknown Face"
        
    best_match_name = "UNKNOWN"
    best_match_id = "Unknown Face"
    min_distance = float("inf")
    
    live_face_resized = cv2.resize(live_face_gray, (50, 50)).astype("float32")
        
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    for folder in folders:
        if "_" not in folder:
            continue
        s_id, s_name = folder.split("_", 1)
        folder_path = os.path.join(dataset_path, folder)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            saved_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if saved_img is not None:
                saved_resized = cv2.resize(saved_img, (50, 50)).astype("float32")
                distance = np.mean((live_face_resized - saved_resized) ** 2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = s_id
                    best_match_name = s_name
                    
    if min_distance < 2500:  
        return best_match_id, best_match_name
    return "UNKNOWN", "Unknown Face"

# --- SAFE FILE HANDLING ---
def load_attendance_data():
    if not os.path.exists("attendance.csv"):
        return pd.DataFrame(columns=["Student ID", "Student Name", "Timestamp"]), "Empty"
    try:
        df = pd.read_csv("attendance.csv")
        if "Student Name" not in df.columns:
            df = pd.DataFrame(columns=["Student ID", "Student Name", "Timestamp"])
        return df, "Success"
    except PermissionError:
        fallback_df = pd.DataFrame([{"Student ID": "⚠️ LOCKED", "Student Name": "Close Excel to fix", "Timestamp": "N/A"}])
        return fallback_df, "Locked"

# --- WebRTC CLOUD VIDEO STREAM WORKERS ---
if WEBRTC_AVAILABLE:
    class CloudLiveAttendanceWorker(VideoTransformerBase):
        def __init__(self):
            self.marked_students = set()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 6)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                current_id, current_name = recognize_face_pure_math(face_roi)
                
                if current_id != "UNKNOWN":
                    color = (16, 185, 129) 
                    label_text = f"{current_name} ({current_id})"
                    
                    if current_id not in self.marked_students:
                        dt_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        try:
                            df, _ = load_attendance_data()
                            new_row = pd.DataFrame([{"Student ID": current_id, "Student Name": current_name, "Timestamp": dt_string}])
                            df = pd.concat([df, new_row], ignore_index=True)
                            df.to_csv("attendance.csv", index=False)
                            self.marked_students.add(current_id)
                        except Exception:
                            pass
                else:
                    color = (239, 68, 68) 
                    label_text = "Unknown Face"
                    
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, label_text, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img

    class CloudRegistrationWorker(VideoTransformerBase):
        def __init__(self, folder_path):
            self.folder_path = folder_path
            self.count = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.count >= 5:
                return img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                if self.count < 5:
                    self.count += 1
                    face_img = gray[y:y+h, x:x+w]
                    cv2.imwrite(f"{self.folder_path}/img_{self.count}.jpg", face_img)
                cv2.rectangle(img, (x, y), (x + w, y + h), (59, 130, 246), 2)
            return img

# Data Pre-Loading for calculation layers
df_logs, file_status = load_attendance_data()
total_registered = len([f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))]) if os.path.exists("dataset") else 0
total_present = df_logs["Student ID"].nunique() if file_status == "Success" else 0

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("### 🕹️ Command Center")
choice = st.sidebar.radio("Navigate System:", ["🏠 Dashboard Hub", "🎥 Live Security Camera Feed", "👤 Register New Profile", "📄 Download Session Report"])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Camera Hardware Routing")
cam_mode = st.sidebar.selectbox("Select Core Camera Engine:", ["🖥️ Local Machine Webcam", "🌐 Live Web Cloud URL"])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Health State:**")
st.sidebar.markdown('<span class="status-badge">● Engine Online (30 FPS)</span>', unsafe_allow_html=True)

st.markdown("---")

# --- 1. DASHBOARD HUB ---
if choice == "🏠 Dashboard Hub":
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f'<div class="metric-box"><p style="color:#64748B; margin-bottom:4px; font-weight:600;">Total Enrolled Users</p><h2 style="color:#1E3A8A; margin:0;">{total_registered} Profiles</h2></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-box"><p style="color:#64748B; margin-bottom:4px; font-weight:600;">Checked-in Today</p><h2 style="color:#10B981; margin:0;">{total_present} Present</h2></div>', unsafe_allow_html=True)
    with m_col3:
        absent_count = max(0, total_registered - total_present)
        st.markdown(f'<div class="metric-box"><p style="color:#64748B; margin-bottom:4px; font-weight:600;">Pending Attendance</p><h2 style="color:#EF4444; margin:0;">{absent_count} Missing</h2></div>', unsafe_allow_html=True)
    with m_col4:
        rate = round((total_present / total_registered * 100), 1) if total_registered > 0 else 0.0
        st.markdown(f'<div class="metric-box"><p style="color:#64748B; margin-bottom:4px; font-weight:600;">Turnout Rate</p><h2 style="color:#F59E0B; margin:0;">{rate}%</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    graph_col, table_col = st.columns([1, 1])
    
    with graph_col:
        st.subheader("📈 Peak Check-In Velocity")
        if file_status == "Success" and not df_logs.empty:
            try:
                df_logs['Hour'] = pd.to_datetime(df_logs['Timestamp']).dt.strftime('%H:00')
                chart_data = df_logs.groupby('Hour').size().reset_index(name='Check-ins')
                st.line_chart(chart_data.set_index('Hour'), color="#3B82F6")
            except Exception:
                st.info("Insufficient timeline data variations to map chart trends yet.")
        else:
            st.info("Line graph will dynamically map arrival spikes as soon as logs roll in.")

    with table_col:
        st.subheader("🔍 Active Attendance Directory Search")
        if file_status == "Locked":
            st.error("🔄 File access blocked. Close 'attendance.csv' in Excel to display records.")
        if file_status == "Success" and not df_logs.empty:
            search_query = st.text_input("Filter spreadsheet by student name or record ID...", placeholder="Type name here...")
            filtered_df = df_logs
            if search_query:
                filtered_df = df_logs[df_logs['Student Name'].str.contains(search_query, case=False, na=False) | df_logs['Student ID'].astype(str).str.contains(search_query, na=False)]
            st.dataframe(filtered_df.sort_values(by="Timestamp", ascending=False), use_container_width=True, height=240)
        else:
            st.info("No logs captured for today's active tracking session yet.")

# --- 2. LIVE SECURITY CAMERA FEED ---
elif choice == "🎥 Live Security Camera Feed":
    st.subheader("📸 High-Frequency Biometric Capture Grid")
    feed_left, feed_right = st.columns([2, 1])
    
    with feed_left:
        if cam_mode == "🖥️ Local Machine Webcam":
            st.caption("Press the toggle below to activate your local computer's built-in camera stream.")
            run_local = st.toggle("Power On Attendance Surveillance Radar", value=False)
            FRAME_WINDOW = st.image([])
            
            if run_local:
                cap = cv2.VideoCapture(0)
                local_marked = set()
                while run_local:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Hardware Alert: Failed to communicate with local camera.")
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.2, 6)
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        current_id, current_name = recognize_face_pure_math(face_roi)
                        if current_id != "UNKNOWN":
                            color = (16, 185, 129)
                            label_text = f"{current_name} ({current_id})"
                            if current_id not in local_marked:
                                dt_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                try:
                                    df, _ = load_attendance_data()
                                    new_row = pd.DataFrame([{"Student ID": current_id, "Student Name": current_name, "Timestamp": dt_string}])
                                    df = pd.concat([df, new_row], ignore_index=True)
                                    df.to_csv("attendance.csv", index=False)
                                    local_marked.add(current_id)
                                except Exception:
                                    pass
                        else:
                            color = (239, 68, 68)
                            label_text = "Unknown Face"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                        cv2.putText(frame, label_text, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    time.sleep(0.01)
                cap.release()
                
        elif cam_mode == "🌐 Live Web Cloud URL":
            if not WEBRTC_AVAILABLE:
                st.error("Package Deployment Error: 'streamlit-webrtc' missing. Check your requirements.txt configuration layer.")
            else:
                st.caption("Activate WebRTC connection stream pipeline to hook system cameras to the cloud.")
                ctx = webrtc_streamer(
                    key="attendance-pipeline",
                    mode=WebRtcMode.SENDRECV,
                    video_transformer_factory=CloudLiveAttendanceWorker,
                    async_transform=True,
                    media_stream_constraints={"video": True, "audio": False}
                )
                
    with feed_right:
        st.markdown("#### ⚡ Terminal Telemetry")
        if cam_mode == "🖥️ Local Machine Webcam":
            st.code("🖥️ Configured for Local Hardware Loops.\nNo browser WebRTC layer required.")
        else:
            if WEBRTC_AVAILABLE and 'ctx' in locals() and ctx.state.playing:
                st.success("🟢 Web Video Pipeline Processing Active")
            else:
                st.code("⚪ Streaming Offline. Connect webcam via start button.")

# --- 3. REGISTER NEW PROFILE ---
elif choice == "👤 Register New Profile":
    st.subheader("📝 Dynamic Identity Registration Array")
    col_a, col_b = st.columns(2)
    
    with col_a:
        student_id = st.text_input("Assign Unique ID (e.g., 101, 102)")
        student_name = st.text_input("Enter Student Full Name")
        st.markdown("<br>", unsafe_allow_html=True)
        start_registration = st.toggle("Power On Matrix Registration Camera", value=False)
        REG_WINDOW = st.image([])
        
    with col_b:
        st.info("💡 **Enrollment Best Practices:** Stand directly in front of the lens. The server catches 5 template iterations securely.")
        
        if start_registration:
            if student_id and student_name:
                folder_path = f"dataset/{student_id.strip()}_{student_name.strip()}"
                os.makedirs(folder_path, exist_ok=True)
                
                if cam_mode == "🖥️ Local Machine Webcam":
                    cap = cv2.VideoCapture(0)
                    count = 0
                    while start_registration and count < 5:
                        ret, frame = cap.read()
                        if not ret: break
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            if count < 5:
                                count += 1
                                face_img = gray[y:y+h, x:x+w]
                                cv2.imwrite(f"{folder_path}/img_{count}.jpg", face_img)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (59, 130, 246), 2)
                        REG_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        time.sleep(0.2)
                    cap.release()
                    if count >= 5: st.success(f"🎉 Saved 5 local templates into '{folder_path}' successfully!")
                    
                elif cam_mode == "🌐 Live Web Cloud URL":
                    if WEBRTC_AVAILABLE:
                        reg_ctx = webrtc_streamer(
                            key=f"reg-stream-{student_id}",
                            mode=WebRtcMode.SENDRECV,
                            video_transformer_factory=lambda: CloudRegistrationWorker(folder_path),
                            async_transform=True,
                            media_stream_constraints={"video": True, "audio": False}
                        )
                    else:
                        st.error("WebRTC modules not found.")
            else:
                st.error("Operation Denied: All biometric identity string fields are strictly required.")

# --- 4.📄 DOWNLOAD SESSION REPORT ---
elif choice == "📄 Download Session Report":
    st.subheader("📄 Automated Operational Document Export")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### Export Active Roster Report")
        if file_status == "Success" and not df_logs.empty:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            report_content = f"===============================================\n"
            report_content += f"      ENTERPRISE ATTENDANCE PORTAL REPORT       \n"
            report_content += f"===============================================\n"
            report_content += f"Generated On: {now_str}\n\n"
            for _, row in df_logs.sort_values(by="Timestamp", ascending=False).iterrows():
                report_content += f"{str(row['Student ID']):<10} | {str(row['Student Name']):<20} | {str(row['Timestamp'])}\n"
            
            st.download_button(
                label="📥 Download Attendance Report (.txt)",
                data=report_content,
                file_name=f"Attendance_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("Cannot compile report: The system log ledger contains 0 entries.")
    with col_r:
        st.info("📌 **Document Export Specifications:**\n\nThe downloaded document aggregates current localized metrics, ready for HR record integration.")
