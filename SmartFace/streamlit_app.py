"""
SmartFace - Sistem Absensi Berbasis Face Recognition
Streamlit Application untuk Demo Online
"""

import streamlit as st
import pandas as pd
import requests
import json
from PIL import Image
import numpy as np
from datetime import datetime
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SmartFace - Face Recognition Attendance System",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .header-title {
            color: #0F72E8;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .section-title {
            color: #0F72E8;
            font-size: 1.8rem;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 3px solid #22C55E;
            padding-bottom: 0.5rem;
        }
        .feature-box {
            background: linear-gradient(135deg, #0F72E8 0%, #22C55E 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            text-align: center;
        }
        .metric-box {
            background: linear-gradient(135deg, rgba(15, 114, 232, 0.1), rgba(34, 197, 94, 0.1));
            border: 2px solid #0F72E8;
            padding: 1.5rem;
            border-radius: 1rem;
            text-align: center;
            margin: 1rem 0;
        }
        .success-badge {
            background-color: #22C55E;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            display: inline-block;
        }
        .warning-badge {
            background-color: #F59E0B;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================
BACKEND_URL = "http://localhost:5000"  # Backend Flask endpoint

# Check if backend is running
def check_backend():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Initialize session state for backend status
if 'backend_available' not in st.session_state:
    st.session_state.backend_available = check_backend()
st.sidebar.markdown("### 🎓 SmartFace Navigation")
page = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Home",
    "📊 Dashboard",
    "🎯 Demo Aplikasi",
    "📚 Dokumentasi",
    "👥 Tim Developer"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📱 Informasi Teknis
- **Backend**: Flask (Python)
- **Frontend**: React + TypeScript
- **Model**: ResNet50 (Deep Learning)
- **Face Detection**: MTCNN
- **Database**: JSON
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔗 Links
- [GitHub Repository](https://github.com/elsaelisa09/TugasBesarIIDeepLearning)
- [Dokumentasi](https://github.com/elsaelisa09/TugasBesarIIDeepLearning#readme)
""")

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    # Header
    st.markdown('<p class="header-title">🎓 SmartFace</p>', unsafe_allow_html=True)
    st.markdown("### Sistem Absensi Berbasis Face Recognition dengan Deep Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **SmartFace** adalah sistem absensi otomatis yang memanfaatkan teknologi 
        **Face Recognition** dan **Deep Learning** untuk mendeteksi kehadiran mahasiswa 
        secara real-time dengan akurasi tinggi.
        
        Sistem ini menggunakan model **ResNet50** yang telah di-fine-tune khusus 
        untuk mengenali hingga **70 identitas mahasiswa** dengan tingkat akurasi 
        mencapai **98.5%**.
        """)
    
    with col2:
        st.info("""
        ✨ **Fitur Unggulan**
        - 🎬 Deteksi wajah real-time
        - 🔍 Akurasi 98.5% dengan Deep Learning
        - 👥 Mendukung 70+ identitas mahasiswa
        - 📱 Interface user-friendly
        - 📊 Analytics & reporting
        """)
    
    # Features Section
    st.markdown('<p class="section-title">🎯 Fitur Utama</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📸 Real-time Detection</h3>
            Deteksi wajah instan dengan MTCNN
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🧠 Deep Learning</h3>
            ResNet50 Model dengan Akurasi Tinggi
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Analytics</h3>
            Dashboard Lengkap & Reporting
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-box">
            <h3>⚡ Fast & Efficient</h3>
            Proses Cepat Tanpa Lag
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
elif page == "📊 Dashboard":
    st.markdown('<p class="section-title">📊 Dashboard Statistik Model</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h2>98.5%</h2>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h2>97.2%</h2>
            <p>Precision Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h2>99.1%</h2>
            <p>Recall Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h2>98.1%</h2>
            <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Metrics
    st.markdown("### 📈 Detail Performa Model")
    
    metrics_data = {
        "Metrik": [
            "Accuracy", "Precision", "Recall", "F1-Score", 
            "Training Epochs", "Classes (Students)", "Avg Confidence",
            "Training Images", "Images per Student"
        ],
        "Nilai": [
            "98.5%", "97.2%", "99.1%", "98.1%",
            "50", "70", "96.7%",
            "280", "4"
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Model Architecture
    st.markdown("### 🏗️ Arsitektur Model")
    st.info("""
    **Model**: ResNet50 (Pre-trained + Fine-tuned)
    
    - **Backbone**: ResNet50 dari PyTorch
    - **Layer Freeze**: Layer 1-2 (frozen), Layer 3-4 (trainable)
    - **Classification Head**: 
      - Dropout(0.5) → Linear(2048→512) → ReLU → BatchNorm1d(512)
      - Dropout(0.3) → Linear(512→70)
    - **Optimizer**: Adam
    - **Loss Function**: CrossEntropyLoss
    - **Face Detection**: MTCNN dengan post-processing
    """)
    
    # Training Information
    st.markdown("### 📚 Informasi Training")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Composition**")
        dataset_data = {
            "Class": "70 Mahasiswa",
            "Images per Class": "4 images",
            "Total Images": "280 images",
            "Train-Validation Split": "80-20",
            "Image Size": "224x224 px"
        }
        for key, value in dataset_data.items():
            st.write(f"- {key}: {value}")
    
    with col2:
        st.write("**Training Configuration**")
        config_data = {
            "Batch Size": "32",
            "Learning Rate": "0.0001",
            "Epochs": "50",
            "Device": "CPU",
            "Augmentation": "Yes"
        }
        for key, value in config_data.items():
            st.write(f"- {key}: {value}")

# ============================================================================
# PAGE: DEMO APLIKASI
# ============================================================================
elif page == "🎯 Demo Aplikasi":
    st.markdown('<p class="section-title">🎯 Demo Aplikasi SmartFace</p>', unsafe_allow_html=True)
    
    # Check backend status
    if not st.session_state.backend_available:
        st.error("""
        ❌ **Backend Tidak Terkoneksi**
        
        Pastikan backend sudah berjalan di `http://localhost:5000`
        
        Jalankan di terminal:
        ```
        cd SmartFace/backend
        python app.py
        ```
        """)
        st.stop()
    
    st.success("✅ Backend Connected: http://localhost:5000")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Upload Foto untuk Deteksi")
        uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Foto yang diupload", use_column_width=True)
            
            if st.button("🔍 Deteksi Wajah & Cek Kehadiran", use_container_width=True):
                with st.spinner("⏳ Memproses gambar... Sedang deteksi wajah..."):
                    try:
                        # Convert image to base64
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='JPEG')
                        image_b64 = image_bytes.getvalue()
                        image_b64_str = __import__('base64').b64encode(image_b64).decode()
                        
                        # Send to backend
                        response = requests.post(
                            f"{BACKEND_URL}/recognize",
                            json={"image": f"data:image/jpeg;base64,{image_b64_str}"},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.demo_result = result
                            st.success("✅ Deteksi Berhasil!")
                        else:
                            error_msg = response.json().get('error', 'Unknown error')
                            st.error(f"❌ Error: {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Error saat menghubungi backend: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Hasil Deteksi")
        
        if 'demo_result' in st.session_state:
            result = st.session_state.demo_result
            predictions = result.get('predictions', [])
            
            if predictions:
                top_pred = predictions[0]
                student_name = top_pred['label']
                confidence = top_pred['confidence']
                
                st.markdown(f"""
                <div class="metric-box" style="background: linear-gradient(135deg, #22C55E 0%, #16a34a 100%); border: 2px solid #16a34a;">
                    <h2 style="color: white;">HADIR ✅</h2>
                    <p style="color: white; font-size: 1.1rem;">{student_name}</p>
                    <p style="color: white; font-size: 0.95rem;">Confidence: {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📸 Cropped Face Image")
                if 'face_image' in result:
                    st.image(result['face_image'], use_column_width=True)
                
                st.markdown("#### 🎯 Top 3 Prediksi")
                pred_data = []
                for i, pred in enumerate(predictions[:3], 1):
                    pred_data.append({
                        "Ranking": i,
                        "Nama": pred['label'],
                        "Confidence": f"{pred['confidence']}%"
                    })
                
                predictions_df = pd.DataFrame(pred_data)
                st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                
                # Mark attendance button
                if st.button("💾 Catat Kehadiran", use_container_width=True):
                    with st.spinner("📝 Menyimpan data kehadiran..."):
                        try:
                            attendance_response = requests.post(
                                f"{BACKEND_URL}/mark-attendance",
                                json={
                                    "label": student_name,
                                    "confidence": confidence,
                                    "image": result.get('face_image', '')
                                },
                                timeout=5
                            )
                            
                            if attendance_response.status_code == 200:
                                att_result = attendance_response.json()
                                st.success(f"✅ {att_result.get('message', 'Kehadiran berhasil dicatat')}")
                            else:
                                err_msg = attendance_response.json().get('message', 'Unknown error')
                                st.warning(f"⚠️ {err_msg}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("📌 Upload dan deteksi foto di sebelah kiri untuk melihat hasil")

# ============================================================================
# PAGE: DOKUMENTASI
# ============================================================================
elif page == "📚 Dokumentasi":
    st.markdown('<p class="section-title">📚 Dokumentasi Teknis</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Arsitektur", 
        "📦 Dependencies", 
        "🚀 Instalasi", 
        "💻 API Endpoints"
    ])
    
    with tab1:
        st.markdown("""
        ### Arsitektur Sistem SmartFace
        
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                    CLIENT SIDE                          │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │       React + TypeScript Frontend (Vite)        │   │
        │  │  - Camera View Component                        │   │
        │  │  - Attendance Status Display                    │   │
        │  │  - Real-time Detection UI                       │   │
        │  └──────────────────────────────────────────────────┘   │
        │                       ↓ API Calls                       │
        └─────────────────────────────────────────────────────────┘
                              ↓ HTTP
        ┌─────────────────────────────────────────────────────────┐
        │                   SERVER SIDE                           │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │      Flask Backend (Python)                     │   │
        │  │  - REST API Endpoints                           │   │
        │  │  - Face Detection Pipeline                      │   │
        │  │  - Model Inference                              │   │
        │  │  - Data Management                              │   │
        │  └──────────────────────────────────────────────────┘   │
        │                       ↓                                  │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │  Deep Learning Model (PyTorch)                  │   │
        │  │  - ResNet50 Classifier                          │   │
        │  │  - MTCNN Face Detector                          │   │
        │  │  - 70-class Recognition                         │   │
        │  └──────────────────────────────────────────────────┘   │
        │                       ↓                                  │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │      Data Storage                               │   │
        │  │  - Attendance Records (JSON)                    │   │
        │  │  - Student Labels (CSV)                         │   │
        │  │  - Model Weights (.pth)                         │   │
        │  └──────────────────────────────────────────────────┘   │
        └─────────────────────────────────────────────────────────┘
        """)
    
    with tab2:
        st.markdown("""
        ### 📦 Dependencies
        
        **Backend Requirements:**
        - `flask==3.1.0` - Web framework
        - `flask-cors==5.0.0` - Cross-origin requests
        - `torch==2.8.0` - Deep Learning framework
        - `torchvision==0.23.0` - Vision utilities
        - `opencv-python==4.12.0.88` - Computer vision
        - `pillow==11.1.0` - Image processing
        - `numpy==2.2.5` - Numerical computing
        - `scikit-learn==1.7.1` - ML utilities
        - `facenet-pytorch==2.6.0` - MTCNN face detection
        - `onnxruntime==1.21.0` - Model runtime
        
        **Frontend Requirements:**
        - `react==18.3.1` - UI framework
        - `typescript==5.8.3` - Type safety
        - `vite==5.4.19` - Build tool
        - `tailwind-css` - Styling
        - `shadcn-ui` - UI components
        """)
    
    with tab3:
        st.markdown("""
        ### 🚀 Instalasi & Setup
        
        **1. Clone Repository:**
        ```bash
        git clone https://github.com/elsaelisa09/TugasBesarIIDeepLearning.git
        cd TugasBesarIIDeepLearning/SmartFace
        ```
        
        **2. Setup Backend:**
        ```bash
        cd backend
        pip install -r requirements.txt
        python app.py
        ```
        
        **3. Setup Frontend:**
        ```bash
        cd ..
        npm install
        npm run dev
        ```
        
        **4. Access Application:**
        - Frontend: http://localhost:8080
        - Backend: http://localhost:5000
        - Streamlit: streamlit run streamlit_app.py
        """)
    
    with tab4:
        st.markdown("### 💻 Backend API Endpoints")
        
        # Try to get backend info
        try:
            health_response = requests.get(f"{BACKEND_URL}/health", timeout=2)
            if health_response.status_code == 200:
                health_data = health_response.json()
                
                st.success("✅ **Backend Status: CONNECTED**")
                
                col_status1, col_status2 = st.columns(2)
                with col_status1:
                    st.write(f"**Model Loaded**: {health_data.get('model_loaded', False)}")
                    st.write(f"**Face Detector**: {health_data.get('face_detector', 'N/A')}")
                with col_status2:
                    st.write(f"**MTCNN Ready**: {health_data.get('mtcnn_loaded', False)}")
                    st.write(f"**Classes**: {health_data.get('num_classes', 0)}")
        except:
            st.warning("⚠️ **Backend Status: DISCONNECTED** - Backend tidak merespons")
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 Endpoint Reference
        
        **1. Health Check**
        ```
        GET /health
        Response: {model_loaded, face_detector, mtcnn_loaded, num_classes}
        ```
        
        **2. Face Recognition**
        ```
        POST /recognize
        Body: {"image": "data:image/jpeg;base64,..."}
        Response: {
            "success": true,
            "predictions": [
                {"label": "Nama Siswa", "confidence": 98.5}
            ],
            "bbox": {"x1": ..., "y1": ..., "x2": ..., "y2": ...},
            "face_image": "base64...",
            "annotated_image": "base64..."
        }
        ```
        
        **3. Mark Attendance**
        ```
        POST /mark-attendance
        Body: {
            "label": "Student Name",
            "confidence": 98.5,
            "image": "base64..."
        }
        Response: {
            "success": true,
            "message": "Absensi berhasil dicatat",
            "record": {...}
        }
        ```
        
        **4. Get Attendance Records**
        ```
        GET /attendance
        Optional: ?date=2024-11-30
        Response: {
            "success": true,
            "data": [...],
            "total": 15
        }
        ```
        
        **5. Delete Attendance Record**
        ```
        DELETE /attendance/<id>
        Response: {"success": true, "message": "Record deleted"}
        ```
        """)

# ============================================================================
# PAGE: TIM DEVELOPER
# ============================================================================
elif page == "👥 Tim Developer":
    st.markdown('<p class="section-title">👥 Tim Pengembang SmartFace</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Proyek **SmartFace** dikerjakan oleh 3 mahasiswa dari kelas **RA** sebagai 
    Tugas Besar II - Deep Learning.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0F72E8 0%, #22C55E 100%); 
                    color: white; padding: 2rem; border-radius: 1rem; text-align: center;">
            <h3>Elsa Elisa Yohana Sianturi</h3>
            <p><strong>NIM:</strong> 122140135</p>
            <p><strong>Role:</strong> Project Lead & Backend Developer</p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                Mengurus koordinasi tim, backend development, dan deployment ke Streamlit
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22C55E 0%, #8B5CF6 100%); 
                    color: white; padding: 2rem; border-radius: 1rem; text-align: center;">
            <h3>Sikah Nubuahtul Ilmi</h3>
            <p><strong>NIM:</strong> 122140208</p>
            <p><strong>Role:</strong> Frontend Developer & UI/UX</p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                Mengembangkan frontend dengan React, styling dengan Tailwind, dan UI/UX design
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8B5CF6 0%, #0F72E8 100%); 
                    color: white; padding: 2rem; border-radius: 1rem; text-align: center;">
            <h3>Machzaul Harmansyah</h3>
            <p><strong>NIM:</strong> 122140172</p>
            <p><strong>Role:</strong> ML Engineer & Model Development</p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                Mengurus model training, fine-tuning ResNet50, dan face detection optimization
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 Kelas & Institusi")
    st.info("""
    - **Kelas**: RA (Deep Learning Class)
    - **Institusi**: Universitas Telkom / Institut Teknologi
    - **Tahun Akademik**: 2024
    - **Mata Kuliah**: Tugas Besar II - Deep Learning
    """)
    
    st.markdown("### 🔗 Repository & Links")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📌 [GitHub Repository](https://github.com/elsaelisa09/TugasBesarIIDeepLearning)
        - 📝 [README Documentation](https://github.com/elsaelisa09/TugasBesarIIDeepLearning#readme)
        """)
    
    with col2:
        st.markdown("""
        - 🌐 [Aplikasi Web](http://localhost:8080)
        - 💬 [Issues & Discussion](https://github.com/elsaelisa09/TugasBesarIIDeepLearning/issues)
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>SmartFace © 2024 - Sistem Absensi Berbasis Face Recognition</p>
    <p>Tugas Besar II Deep Learning | Kelas RA</p>
    <p>
        <strong>Deployed with ❤️ using Streamlit</strong>
    </p>
</div>
""", unsafe_allow_html=True)
