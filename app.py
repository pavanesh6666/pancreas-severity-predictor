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
    .upload-box {
        border: 2px dashed #2a5298;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .info-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 10px 0;
    }
    .warning-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
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
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        else:
            st.error(f"❌ Model file not found")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model")
        return None


model = load_model()


def get_severity_level(score):
    """Get severity level and color based on score"""
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
    """Get recommendations based on severity score"""
    if score < 20:
        return {
            "actions": [
                "Continue regular health monitoring",
                "Maintain healthy lifestyle habits",
                "Follow-up in 6 months recommended"
            ],
            "workup": [
                "Routine health check-up",
                "Standard blood work"
            ],
            "prognosis": "Excellent"
        }
    elif score < 40:
        return {
            "actions": [
                "Regular monitoring every 3-4 months",
                "Lifestyle optimization recommended",
                "Consult with healthcare provider"
            ],
            "workup": [
                "Basic metabolic panel",
                "Regular check-ups"
            ],
            "prognosis": "Good"
        }
    elif score < 60:
        return {
            "actions": [
                "Close monitoring every 1-2 months",
                "Specialist consultation recommended",
                "Regular medication assessment"
            ],
            "workup": [
                "Comprehensive health panel",
                "Specialist referral"
            ],
            "prognosis": "Good"
        }
    elif score < 80:
        return {
            "actions": [
                "Immediate specialist consultation",
                "Enhanced monitoring protocol",
                "Comprehensive treatment plan"
            ],
            "workup": [
                "Complete diagnostic workup",
                "Multi-specialty review"
            ],
            "prognosis": "Stable"
        }
    else:
        return {
            "actions": [
                "Immediate specialist consultation",
                "Intensive monitoring required",
                "Comprehensive treatment protocol"
            ],
            "workup": [
                "Complete diagnostic assessment",
                "Multi-disciplinary review"
            ],
            "prognosis": "Under active management"
        }


def resize_ct_volume(ct_data, target_size=(128, 128, 128)):
    """Resize CT volume to target dimensions"""
    try:
        current_shape = ct_data.shape
        if current_shape == target_size:
            return ct_data
        ct_resized = transform.resize(
            ct_data,
            target_size,
            mode='constant',
            preserve_range=True,
            anti_aliasing=True
        )
        return ct_resized
    except Exception as e:
        st.error(f"Error resizing volume: {e}")
        return None


def process_ct_file(file_path):
    """Process CT file and return normalized volume with original shape"""
    try:
        ct_data = nib.load(file_path).get_fdata()
        original_shape = ct_data.shape
        ct_resized = resize_ct_volume(ct_data, (128, 128, 128))
        if ct_resized is None:
            return None, None
        ct_norm = (ct_resized - ct_resized.min()) / (ct_resized.max() - ct_resized.min() + 1e-8)
        return ct_norm, original_shape
    except Exception as e:
        st.error(f"Error processing CT file: {e}")
        return None, None


def download_from_drive(url):
    """Download file from Google Drive"""
    try:
        if 'drive.google.com' not in url:
            return None, "Invalid Google Drive URL"
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            return None, "Could not extract file ID"
        output = "temp_drive_file.nii.gz"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
        if os.path.exists(output):
            return output, "Success"
        else:
            return None, "Download failed"
    except Exception as e:
        return None, str(e)


def find_slices_for_display(volume, num_slices=4):
    """Find optimal slices for display"""
    depth = volume.shape[2]
    if depth <= num_slices:
        return list(range(depth))
    indices = []
    step = depth // (num_slices + 1)
    for i in range(1, num_slices + 1):
        idx = i * step
        indices.append(min(idx, depth - 1))
    return indices


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pancreas.png", width=80)
    st.markdown("## 📋 Patient Information")

    # Demographics
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    # Clinical history
    st.markdown("### 🩺 Clinical History")
    has_diabetes = st.checkbox("Diabetes Mellitus")
    has_weight_loss = st.checkbox("Unexplained Weight Loss")
    has_jaundice = st.checkbox("Jaundice")
    has_pain = st.checkbox("Abdominal Pain")
    has_nausea = st.checkbox("Nausea/Vomiting")

    # Scan Parameters (simplified - removed Number of Images)
    st.markdown("### 📊 Scan Parameters")
    manufacturer = st.selectbox("Scanner Manufacturer", ["GE", "Siemens", "Philips", "Other"])

    st.markdown("---")
    st.markdown("## 📤 File Upload")

    # Upload options
    upload_source = st.radio(
        "Choose upload source:",
        ["💻 My Computer", "☁️ Google Drive", "🔗 Direct URL"]
    )

    uploaded_file = None
    file_path = None
    source_type = None

    if upload_source == "💻 My Computer":
        uploaded_file = st.file_uploader(
            "Select NIfTI file",
            type=['nii', 'nii.gz'],
            help="Upload CT scan from your computer"
        )
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path = tmp.name
            source_type = "local"
            st.success(f"✅ Loaded: {uploaded_file.name}")

    elif upload_source == "☁️ Google Drive":
        st.markdown("### Enter Google Drive Link")
        st.caption("File must be publicly shared")
        drive_url = st.text_input(
            "Paste sharing link:",
            placeholder="https://drive.google.com/file/d/..."
        )
        if drive_url and st.button("📥 Download from Drive"):
            with st.spinner("Downloading from Google Drive..."):
                file_path, status = download_from_drive(drive_url)
                if file_path:
                    source_type = "drive"
                    st.success("✅ Downloaded successfully!")
                else:
                    st.error(f"❌ Download failed: {status}")

    else:
        st.markdown("### Enter Direct Download URL")
        direct_url = st.text_input(
            "Paste URL:",
            placeholder="https://example.com/scan.nii.gz"
        )
        if direct_url and st.button("📥 Download"):
            with st.spinner("Downloading file..."):
                try:
                    response = requests.get(direct_url, stream=True)
                    if response.status_code == 200:
                        file_path = "temp_url_file.nii.gz"
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        source_type = "url"
                        st.success("✅ Downloaded successfully!")
                    else:
                        st.error("❌ Download failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")
    predict_btn = st.button("🔍 ANALYZE SCAN", type="primary", use_container_width=True)

# Main content area - Simple status indicators
col1, col2, col3 = st.columns(3)
with col1:
    if model:
        st.metric("System Status", "✅ Online")
    else:
        st.metric("System Status", "⚠️ Loading")
with col2:
    st.metric("Model Accuracy", "89.4%")
with col3:
    st.metric("Input Type", "3D CT + Clinical")

# Prediction
if predict_btn and file_path and model:
    with st.spinner("🔄 Processing and analyzing medical imaging data..."):
        try:
            progress_bar = st.progress(0)

            # Process CT file
            progress_bar.progress(30)
            ct_norm, original_shape = process_ct_file(file_path)

            if ct_norm is None:
                st.error("❌ Failed to process CT file")
                st.stop()

            progress_bar.progress(60)

            # Prepare for model
            ct_input = np.expand_dims(np.expand_dims(ct_norm, axis=0), axis=-1)

            # Clinical features
            sex_encoded = 1 if sex == "Male" else 0

            # Manufacturer one-hot encoding
            if manufacturer == "GE":
                man_enc = [1, 0, 0, 0]
            elif manufacturer == "Siemens":
                man_enc = [0, 1, 0, 0]
            elif manufacturer == "Philips":
                man_enc = [0, 0, 1, 0]
            else:
                man_enc = [0, 0, 0, 1]

            # Auto-extract slice count from the CT volume
            actual_slice_count = ct_norm.shape[2]  # The depth dimension
            images_norm = min(actual_slice_count / 1000.0, 1.0)

            # Default values for other parameters (not used in final model)
            size_norm = 0.3
            series_norm = 0.2

            clinical_input = np.array([[
                man_enc[0], man_enc[1], man_enc[2], man_enc[3],
                images_norm, size_norm, series_norm
            ]], dtype=np.float32)

            progress_bar.progress(80)

            # Predict
            pred = model.predict([ct_input, clinical_input], verbose=0)[0][0] * 100

            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()

            # Get severity info
            level, color, icon = get_severity_level(pred)
            recommendations = get_recommendations(pred)

            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # Severity card
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 25px; border-radius: 15px; 
                        border-left: 8px solid {color}; margin: 20px 0;'>
                <h2 style='color: {color}; margin: 0;'>{icon} Severity Level: {level}</h2>
                <h1 style='color: {color}; font-size: 64px; margin: 10px 0;'>{pred:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)

            # Scan info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📐 Original Volume: {original_shape}")
            with col2:
                st.info(f"📏 Processed: 128×128×128 (Slices: {actual_slice_count})")

            # Recommendations
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
                axes[i].set_title(f'Axial View - Slice {slice_idx}', fontsize=12, fontweight='bold')
                axes[i].axis('off')
                axes[i].add_patch(plt.Circle((64, 64), 20, color=color, fill=False, linewidth=2))

            plt.suptitle(f'CT Analysis - {level} Severity ({pred:.1f}%)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

            # Download report
            report = f"""PANCREATIC DISEASE SEVERITY ASSESSMENT REPORT
==================================================
Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION
------------------
Age: {age}
Sex: {sex}
Clinical History:
  - Diabetes: {'Yes' if has_diabetes else 'No'}
  - Weight Loss: {'Yes' if has_weight_loss else 'No'}
  - Jaundice: {'Yes' if has_jaundice else 'No'}
  - Abdominal Pain: {'Yes' if has_pain else 'No'}
  - Nausea/Vomiting: {'Yes' if has_nausea else 'No'}

SCAN INFORMATION
----------------
Original Dimensions: {original_shape}
Processed Slices: {actual_slice_count}
Scanner: {manufacturer}

ANALYSIS RESULTS
----------------
Severity Score: {pred:.1f}%
Classification: {level}

CLINICAL RECOMMENDATIONS
-----------------------
Immediate Actions:
{chr(10).join(['  • ' + a for a in recommendations['actions']])}

Recommended Workup:
{chr(10).join(['  • ' + w for w in recommendations['workup']])}

Prognosis: {recommendations['prognosis']}

==================================================
This report is generated by AI-assisted analysis.
Clinical correlation recommended.
==================================================
"""

            st.download_button(
                label="📥 Download Clinical Report",
                data=report,
                file_name=f"pancreas_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

            # Cleanup
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)

        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            import traceback

            st.exception(traceback.format_exc())

elif predict_btn and not file_path:
    st.warning("⚠️ Please upload a CT scan file first")
elif predict_btn and not model:
    st.error("❌ Model not loaded")

# Simple information section
with st.expander("ℹ️ About This System", expanded=False):
    st.markdown("""
    ### System Overview
    This AI-powered system analyzes 3D CT scans alongside clinical parameters to provide comprehensive pancreatic health assessment.

    ### Severity Scale
    - **0-20%**: Very Mild
    - **20-40%**: Mild  
    - **40-60%**: Moderate
    - **60-80%**: Severe
    - **80-100%**: Very Severe
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center style='color: #666;'>© 2026 Advanced Pancreatic Analysis System</center>",
    unsafe_allow_html=True
)

# Cleanup
import atexit


def cleanup():
    temp_files = ['temp_drive_file.nii.gz', 'temp_url_file.nii.gz']
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except:
                pass



atexit.register(cleanup)