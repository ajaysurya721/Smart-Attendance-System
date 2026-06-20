import streamlit as st
import cv2
import pandas as pd
import os
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

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

st.markdown('<p class="main-title">🛡️ AI Enterprise Smart Attendance Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated real-time cloud facial analytics, logging pipeline, and workforce insights dashboard.</p>', unsafe_allow_html=True)

# Initialize standard OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- OPTIMIZED PIXEL MATCHING CORE ENGINE ---
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
        return df, "Success"
    except Exception:
        return pd.DataFrame(columns=["Student ID", "Student Name", "Timestamp"]), "Error"

# --- WebRTC CLOUD VIDEO TRANSFORMER ---
class CloudFaceRecognizer(VideoTransformerBase):
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
                color = (16, 185, 129) # Green
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
                color = (68, 68, 239) # Red
                label_text = "Unknown Face"
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label_text, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return img

# Pre-Load Metrics
df_logs, file_status = load_attendance_data()
total_registered = len([f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))]) if os.path.exists("dataset") else 0
total_present = df_logs["Student ID"].nunique() if file_status == "Success" else 0

# --- SIDEBAR NAV ---
st.sidebar.markdown("### 🕹️ Command Center")
choice = st.sidebar.radio("Navigate System:", ["🏠 Dashboard Hub", "🎥 Live Security Camera Feed", "👤 Register New Profile", "📄 Download Session Report"])
st.sidebar.markdown("---")
st.sidebar.markdown('<span class="status-badge">● Cloud Engine Online</span>', unsafe_allow_html=True)

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
    st.subheader("🔍 Active Attendance Directory Search")
    st.dataframe(df_logs.sort_values(by="Timestamp", ascending=False), use_container_width=True, height=300)

# --- 2. LIVE SECURITY CAMERA FEED ---
elif choice == "🎥 Live Security Camera Feed":
    st.subheader("📸 High-Frequency Browser Biometric Stream")
    st.caption("Click 'Start' below to authorize web camera feed processing directly into the analytics engine layer.")
    
    # WebRTC stream component handles cross-origin browser cameras safely in cloud
    webrtc_streamer(
        key="attendance-stream",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=CloudFaceRecognizer,
        async_transform=True,
        media_stream_constraints={"video": True, "audio": False}
    )

# --- 3. REGISTER NEW PROFILE ---
elif choice == "👤 Register New Profile":
    st.subheader("重新 Dynamic Identity Registration Array")
    st.warning("⚠️ Local file system writes are optimized for local deployment. For cloud profile creation, maintain the root 'dataset/' folder structure in your source repository code tree.")
    
    student_id = st.text_input("Assign Unique ID (e.g., 101)")
    student_name = st.text_input("Enter Student Full Name")
    if st.button("Verify Profile Config Layout"):
        if student_id and student_name:
            st.info(f"Target profile node allocated: dataset/{student_id}_{student_name}/")
        else:
            st.error("Fields are strictly required.")

# --- 4.📄 DOWNLOAD SESSION REPORT ---
elif choice == "📄 Download Session Report":
    st.subheader("📄 Automated Operational Document Export")
    if file_status == "Success" and not df_logs.empty:
        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Attendance CSV Log Spreadsheet",
            data=csv_data,
            file_name=f"Cloud_Attendance_Log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("Ledger contains 0 active records to output.")
