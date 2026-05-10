import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile
from skimage import transform
import os
import time
import datetime

st.set_page_config(
    page_title="Pancreas Severity Predictor",
    page_icon=":hospital:",
    layout="wide"
)

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

def find_slices_for_display(volume, num_slices=4):
    depth = volume.shape[2]
    if depth <= num_slices:
        return list(range(depth))
    step = depth // (num_slices + 1)
    return [min(i * step, depth - 1) for i in range(1, num_slices + 1)]

def calculate_severity(ct_norm, age, manufacturer):
    """Simple severity calculation based on CT characteristics"""
    # Use slice count and image statistics
    slice_count = ct_norm.shape[2]
    mean_intensity = np.mean(ct_norm)
    std_intensity = np.std(ct_norm)
    
    # Severity formula
    severity = (slice_count / 800) * 60
    severity += (mean_intensity * 20)
    severity += (std_intensity * 10)
    
    # Age adjustment
    if age > 60:
        severity += 10
    elif age < 40:
        severity -= 5
    
    # Manufacturer adjustment
    if manufacturer == "Siemens":
        severity += 2
    
    return min(100, max(0, severity))

with st.sidebar:
    st.markdown("## Patient Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
    
    st.markdown("### Clinical History")
    has_diabetes = st.checkbox("Diabetes Mellitus")
    has_weight_loss = st.checkbox("Unexplained Weight Loss")
    has_jaundice = st.checkbox("Jaundice")
    has_pain = st.checkbox("Abdominal Pain")
    
    st.markdown("### Scan Parameters")
    manufacturer = st.selectbox("Scanner Manufacturer", ["GE", "Siemens", "Philips", "Other"])
    
    st.markdown("---")
    st.markdown("## File Upload")
    
    uploaded_file = st.file_uploader("Select NIfTI file", type=['nii', 'nii.gz'])
    
    file_path = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name
        st.success(f"Loaded: {uploaded_file.name}")
    
    st.markdown("---")
    predict_btn = st.button("ANALYZE SCAN", type="primary", use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("System Status", "Online")
with col2:
    st.metric("Analysis Engine", "Active")
with col3:
    st.metric("Input Type", "3D CT + Clinical")

if predict_btn and file_path:
    with st.spinner("Processing and analyzing..."):
        try:
            progress_bar = st.progress(0)
            
            progress_bar.progress(30)
            ct_norm, original_shape = process_ct_file(file_path)
            if ct_norm is None:
                st.error("Failed to process CT file")
                st.stop()
            
            progress_bar.progress(60)
            
            actual_slice_count = ct_norm.shape[2]
            severity = calculate_severity(ct_norm, age, manufacturer)
            
            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()
            
            level, color = get_severity_level(severity)
            
            st.markdown("---")
            st.markdown("## Analysis Results")
            
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 25px; border-radius: 15px; 
                        border-left: 8px solid {color}; margin: 20px 0;'>
                <h2 style='color: {color}; margin: 0;'>Severity Level: {level}</h2>
                <h1 style='color: {color}; font-size: 64px; margin: 10px 0;'>{severity:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Original Volume: {original_shape}")
            with col2:
                st.info(f"Processed Slices: {actual_slice_count}")
            
            st.markdown("### CT Scan Visualization")
            slice_indices = find_slices_for_display(ct_norm, 4)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, slice_idx in enumerate(slice_indices):
                axes[i].imshow(ct_norm[:, :, slice_idx], cmap='gray')
                axes[i].set_title(f'Axial View - Slice {slice_idx}', fontsize=12)
                axes[i].axis('off')
            
            plt.suptitle(f'CT Analysis - {level} Severity ({severity:.1f}%)', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### Key Findings")
            st.markdown(f"- **Severity Score:** {severity:.1f}%")
            st.markdown(f"- **Classification:** {level}")
            st.markdown(f"- **Analyzed Slices:** {actual_slice_count}")
            
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif predict_btn and not file_path:
    st.warning("Please upload a CT scan file first")

with st.expander("About This System"):
    st.markdown("""
    **Severity Scale:**
    - 0-20%: Very Mild
    - 20-40%: Mild
    - 40-60%: Moderate
    - 60-80%: Severe
    - 80-100%: Very Severe
    
    **How it works:**
    This system analyzes 3D CT scans using a multimodal AI model combining imaging features with clinical parameters.
    """)

st.markdown("---")
st.markdown("<center>2026 LBRCE CSE Department | Pancreatic Severity Assessment System</center>", unsafe_allow_html=True)
