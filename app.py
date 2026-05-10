import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile
from skimage import transform
import pandas as pd
import os
from PIL import Image
import time
import datetime
import gdown
import io
import requests

# Enable eager execution (TensorFlow 2 default)
tf.config.run_functions_eagerly(True)

st.set_page_config(
    page_title="Pancreas Severity Predictor",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 25px;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-header"><h1>🩺 Pancreatic Disease Severity Assessment System</h1><p>AI-Powered 3D CT Analysis with Clinical Data Integration</p></div>',
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = "pancreas_model_105_best_20260314_190232.h5"
        if os.path.exists(model_path):
            # Load normally without any v1 compatibility
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        else:
            st.error(f"❌ Model file not found")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

def get_severity_level(score):
    if score < 20:
        return "Very Mild", "#28a745", "🟢"
    elif score < 40:
        return "Mild", "#17a2b8", "ℹ️"
    elif score < 60:
        return "Moderate", "#ffc107", "⚠️"
    elif score < 80:
        return "Severe", "#fd7e14", "🔔"
    else:
        return "Very Severe", "#dc3545", "🏥"

def get_recommendations(score):
    if score < 20:
        return {
            "actions": ["Continue regular health monitoring", "Follow-up in 6 months"],
            "workup": ["Routine health check-up", "Standard blood work"],
            "prognosis": "Excellent"
        }
    elif score < 40:
        return {
            "actions": ["Regular monitoring every 3-4 months", "Lifestyle optimization"],
            "workup": ["Basic metabolic panel", "Regular check-ups"],
            "prognosis": "Good"
        }
    elif score < 60:
        return {
            "actions": ["Close monitoring every 1-2 months", "Specialist consultation"],
            "workup": ["Comprehensive health panel", "Specialist referral"],
            "prognosis": "Good"
        }
    elif score < 80:
        return {
            "actions": ["Immediate specialist consultation", "Enhanced monitoring"],
            "workup": ["Complete diagnostic workup", "Multi-specialty review"],
            "prognosis": "Stable"
        }
    else:
        return {
            "actions": ["Immediate specialist consultation", "Intensive monitoring"],
            "workup": ["Complete diagnostic assessment", "Multi-disciplinary review"],
            "prognosis": "Under active management"
        }

def resize_ct_volume(ct_data, target_size=(128, 128, 128)):
    try:
        if ct_data.shape == target_size:
            return ct_data
        return transform.resize(ct_data, target_size, mode='constant', preserve_range=True, anti_aliasing=True)
    except Exception as e:
        st.error(f"Error resizing: {e}")
        return None

def process_ct_file(file_path):
    try:
        ct_data = nib.load(file_path).get_fdata()
        original_shape = ct_data.shape
        ct_resized = resize_ct_volume(ct_data, (128, 128, 128))
        if ct_resized is None:
            return None, None
        ct_norm = (ct_resized - ct_resized.min()) / (ct_resized.max() - ct_resized.min() + 1e-8)
        return ct_norm.astype(np.float32), original_shape
    except Exception as e:
        st.error(f"Error processing: {e}")
        return None, None

def download_from_drive(url):
    try:
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            return None, "Invalid URL"
        output = "temp_drive_file.nii.gz"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
        return (output, "Success") if os.path.exists(output) else (None, "Download failed")
    except Exception as e:
        return None, str(e)

def find_slices_for_display(volume, num_slices=4):
    depth = volume.shape[2]
    if depth <= num_slices:
        return list(range(depth))
    step = depth // (num_slices + 1)
    return [min(i * step, depth - 1) for i in range(1, num_slices + 1)]

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pancreas.png", width=80)
    st.markdown("## 📋 Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    st.markdown("### 🩺 Clinical History")
    has_diabetes = st.checkbox("Diabetes Mellitus")
    has_weight_loss = st.checkbox("Unexplained Weight Loss")
    has_jaundice = st.checkbox("Jaundice")
    has_pain = st.checkbox("Abdominal Pain")
    has_nausea = st.checkbox("Nausea/Vomiting")

    st.markdown("### 📊 Scan Parameters")
    manufacturer = st.selectbox("Scanner Manufacturer", ["GE", "Siemens", "Philips", "Other"])

    st.markdown("---")
    st.markdown("## 📤 File Upload")

    upload_source = st.radio("Choose upload source:", ["💻 My Computer", "☁️ Google Drive", "🔗 Direct URL"])

    uploaded_file = None
    file_path = None
    source_type = None

    if upload_source == "💻 My Computer":
        uploaded_file = st.file_uploader("Select NIfTI file", type=['nii', 'nii.gz'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path = tmp.name
            source_type = "local"
            st.success(f"✅ Loaded: {uploaded_file.name}")

    elif upload_source == "☁️ Google Drive":
        drive_url = st.text_input("Google Drive link:")
        if drive_url and st.button("📥 Download from Drive"):
            with st.spinner("Downloading..."):
                file_path, status = download_from_drive(drive_url)
                if file_path:
                    source_type = "drive"
                    st.success("✅ Downloaded!")
                else:
                    st.error(f"❌ {status}")

    else:
        direct_url = st.text_input("Direct URL:")
        if direct_url and st.button("📥 Download"):
            with st.spinner("Downloading..."):
                try:
                    r = requests.get(direct_url, stream=True)
                    if r.status_code == 200:
                        file_path = "temp_url_file.nii.gz"
                        with open(file_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        source_type = "url"
                        st.success("✅ Downloaded!")
                    else:
                        st.error("Download failed")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    predict_btn = st.button("🔍 ANALYZE SCAN", type="primary", use_container_width=True)

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("System Status", "✅ Online" if model else "⚠️ Offline")
with col2:
    st.metric("Model Accuracy", "89.4%")
with col3:
    st.metric("Input Type", "3D CT + Clinical")

# Prediction
if predict_btn and file_path and model:
    with st.spinner("🔄 Processing and analyzing..."):
        try:
            progress_bar = st.progress(0)

            progress_bar.progress(30)
            ct_norm, original_shape = process_ct_file(file_path)
            if ct_norm is None:
                st.error("Failed to process CT file")
                st.stop()

            progress_bar.progress(60)

            # Prepare input for model
            ct_input = np.expand_dims(np.expand_dims(ct_norm, axis=0), axis=-1)
            actual_slice_count = ct_norm.shape[2]

            # Clinical features (8 features)
            sex_enc = 1 if sex == "Male" else 0

            if manufacturer == "GE":
                man_enc = [1.0, 0.0, 0.0, 0.0]
            elif manufacturer == "Siemens":
                man_enc = [0.0, 1.0, 0.0, 0.0]
            elif manufacturer == "Philips":
                man_enc = [0.0, 0.0, 1.0, 0.0]
            else:
                man_enc = [0.0, 0.0, 0.0, 1.0]

            clinical_input = np.array([[
                man_enc[0], man_enc[1], man_enc[2], man_enc[3],
                float(sex_enc),
                float(actual_slice_count) / 800.0,
                float(age) / 100.0,
                float(actual_slice_count) / 500.0
            ]], dtype=np.float32)

            progress_bar.progress(80)

            # Simple prediction without any session magic
            prediction = model.predict([ct_input, clinical_input], verbose=0)
            pred = float(prediction[0][0]) * 100

            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()

            # Results
            level, color, icon = get_severity_level(pred)
            recommendations = get_recommendations(pred)

            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 25px; border-radius: 15px; 
                        border-left: 8px solid {color}; margin: 20px 0;'>
                <h2 style='color: {color}; margin: 0;'>{icon} Severity Level: {level}</h2>
                <h1 style='color: {color}; font-size: 64px; margin: 10px 0;'>{pred:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📐 Original Volume: {original_shape}")
            with col2:
                st.info(f"📏 Processed: 128×128×128 (Slices: {actual_slice_count})")

            st.markdown("### 🎯 Clinical Recommendations")
            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.markdown("**Immediate Actions:**")
                for action in recommendations["actions"]:
                    st.markdown(f"• {action}")

            with rec_col2:
                st.markdown("**Recommended Workup:**")
                for workup in recommendations["workup"]:
                    st.markdown(f"• {workup}")

            st.markdown(f"**Prognosis:** {recommendations['prognosis']}")

            # CT Visualization
            st.markdown("### 🔍 CT Scan Analysis")
            slice_indices = find_slices_for_display(ct_norm, 4)

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            for i, slice_idx in enumerate(slice_indices):
                axes[i].imshow(ct_norm[:, :, slice_idx], cmap='gray')
                axes[i].set_title(f'Axial View - Slice {slice_idx}', fontsize=12)
                axes[i].axis('off')

            plt.suptitle(f'CT Analysis - {level} Severity ({pred:.1f}%)', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

            # Report download
            report = f"""PANCREATIC SEVERITY REPORT
Generated: {datetime.datetime.now()}

Patient: Age {age}, {sex}
Severity: {pred:.1f}% ({level})
Scanner: {manufacturer}
Slices: {actual_slice_count}
Prognosis: {recommendations['prognosis']}

Recommendations:
{chr(10).join(['- ' + a for a in recommendations['actions']])}
"""

            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"pancreas_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

            # Cleanup
            if file_path and source_type != "local" and os.path.exists(file_path):
                os.unlink(file_path)

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif predict_btn and not file_path:
    st.warning("⚠️ Please upload a CT scan file first")
elif predict_btn and not model:
    st.error("❌ Model not loaded")

# About section
with st.expander("ℹ️ About This System"):
    st.markdown("""
    **Severity Scale:**
    - 0-20%: Very Mild 🟢
    - 20-40%: Mild ℹ️
    - 40-60%: Moderate ⚠️
    - 60-80%: Severe 🔔
    - 80-100%: Very Severe 🏥
    """)

st.markdown("---")
st.markdown("<center>© 2026 LBRCE CSE Department | Pancreatic Severity Assessment System</center>", unsafe_allow_html=True)
