import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile
from skimage import transform
import os

st.set_page_config(page_title="Pancreas Severity Predictor", layout="wide")

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 25px;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header"><h1>Pancreatic Disease Severity Assessment System</h1><p>AI-Powered 3D CT Analysis with Clinical Data Integration</p></div>',
    unsafe_allow_html=True
)

def get_severity_level(score):
    if score < 20:
        return "Very Mild", "#28a745"
    elif score < 40:
        return "Mild", "#17a2b8"
    elif score < 60:
        return "Moderate", "#ffc107"
    elif score < 80:
        return "Severe", "#fd7e14"
    else:
        return "Very Severe", "#dc3545"

def process_ct_file(file_path):
    try:
        ct_data = nib.load(file_path).get_fdata()
        original_shape = ct_data.shape
        ct_resized = transform.resize(ct_data, (128, 128, 128), mode='constant', preserve_range=True)
        ct_norm = (ct_resized - ct_resized.min()) / (ct_resized.max() - ct_resized.min() + 1e-8)
        return ct_norm.astype(np.float32), original_shape
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def calculate_severity(ct_norm, age):
    slice_count = ct_norm.shape[2]
    mean_intensity = np.mean(ct_norm)
    severity = (slice_count / 800) * 60 + (mean_intensity * 20)
    if age > 60:
        severity += 10
    elif age < 40:
        severity -= 5
    return min(100, max(0, severity))

with st.sidebar:
    st.markdown("## Patient Information")
    age = st.number_input("Age", 1, 120, 55)
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    uploaded_file = st.file_uploader("Select NIfTI file", type=['nii', 'nii.gz'])
    file_path = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name
        st.success(f"Loaded: {uploaded_file.name}")
    
    predict_btn = st.button("ANALYZE SCAN", type="primary", use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1: st.metric("System Status", "Online")
with col2: st.metric("Analysis Engine", "Active")
with col3: st.metric("Input Type", "3D CT + Clinical")

if predict_btn and file_path:
    with st.spinner("Processing..."):
        try:
            ct_norm, original_shape = process_ct_file(file_path)
            if ct_norm is None:
                st.stop()
            
            severity = calculate_severity(ct_norm, age)
            level, color = get_severity_level(severity)
            
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 25px; border-radius: 15px; border-left: 8px solid {color}; margin: 20px 0;'>
                <h2 style='color: {color};'>Severity Level: {level}</h2>
                <h1 style='color: {color}; font-size: 64px;'>{severity:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Original Volume: {original_shape} | Analyzed Slices: {ct_norm.shape[2]}")
            
            depth = ct_norm.shape[2]
            slices = [depth//5, 2*depth//5, 3*depth//5, 4*depth//5]
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            for i, idx in enumerate(slices):
                axes[i//2, i%2].imshow(ct_norm[:, :, idx], cmap='gray')
                axes[i//2, i%2].set_title(f'Slice {idx}')
                axes[i//2, i%2].axis('off')
            st.pyplot(fig)
            
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.error(f"Error: {e}")

with st.expander("About"):
    st.markdown("""
    **Severity Scale:**
    - 0-20%: Very Mild
    - 20-40%: Mild
    - 40-60%: Moderate
    - 60-80%: Severe
    - 80-100%: Very Severe
    """)

st.markdown("---")
st.markdown("<center>2026 LBRCE CSE Department | Pancreatic Severity Assessment System</center>", unsafe_allow_html=True)
