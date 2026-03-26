import sys
import os

# Ensure the app can find the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.inference import predict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Detector Pro",
    page_icon="🧬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    h1, h2, h3 { color: #e2e8f0; }
    .stButton>button {
        background-color: #22c55e;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        height: 3em;
    }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; }
    div[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🧬 Breast Cancer Detection Analysis")
st.markdown("Advanced Machine Learning diagnostic tool for tumor classification.")
st.warning("⚠️ **Disclaimer:** Educational purposes only. Not a medical substitute.")

# ---------------- SESSION STATE ----------------
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'input_df' not in st.session_state:
    st.session_state.input_df = None

# ---------------- INPUT SECTION ----------------
st.markdown("### 🧾 Patient Clinical Data")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📏 Size & Shape")
        mean_radius = st.slider('Mean Radius', 6.0, 30.0, 12.0)
        mean_area = st.slider('Mean Area', 150.0, 2500.0, 500.0)
        mean_peri = st.slider('Mean Perimeter', 40.0, 190.0, 80.0)
        area_error = st.slider('Area Error', 5.0, 550.0, 50.0)
        worst_rad = st.slider('Worst Radius', 7.0, 35.0, 15.0)

    with col2:
        st.subheader("🧪 Texture & Concavity")
        mean_concavity = st.slider('Mean Concavity', 0.0, 0.45, 0.10)
        mean_concave = st.slider('Mean Concave Points', 0.0, 0.20, 0.05)
        worst_peri = st.slider('Worst Perimeter', 50.0, 260.0, 100.0)
        worst_area = st.slider("Worst Area", 200.0, 4500.0, 500.0)
        worst_concave = st.slider('Worst Concave Points', 0.0, 0.30, 0.10)

# ---------------- PREDICTION LOGIC ----------------
st.markdown("---")
if st.button("🚀 Run Comprehensive Diagnostic Analysis"):
    # Feature columns MUST match the model's training order exactly
    feature_columns = [
        'worst perimeter', 'worst area', 'worst radius', 'mean concave points',
        'worst concave points', 'mean perimeter', 'mean concavity',
        'mean radius', 'mean area', 'area error'
    ]
    
    values = [
        worst_peri, worst_area, worst_rad, mean_concave, 
        worst_concave, mean_peri, mean_concavity, 
        mean_radius, mean_area, area_error
    ]

    input_df = pd.DataFrame([values], columns=feature_columns)
    
    # Run prediction
    pred, proba = predict(input_df)

    st.session_state.prediction = pred
    st.session_state.confidence = proba
    st.session_state.input_df = input_df

# ---------------- RESULTS DISPLAY ----------------
if st.session_state.prediction is not None:
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("🔍 Analysis Result")
        if st.session_state.prediction == 1:
            score = st.session_state.confidence[1] * 100
            st.success(f"### **BENIGN**\nLow Risk Profile")
            st.metric("Confidence", f"{score:.2f}%")
        else:
            score = st.session_state.confidence[0] * 100
            st.error(f"### **MALIGNANT**\nHigh Risk Profile")
            st.metric("Confidence", f"{score:.2f}%")

    # ---------------- COMPREHENSIVE VISUALIZATION ----------------
    st.markdown("### 📊 All Feature Threshold Analysis")
    
    # Clinical Thresholds for all 10 features
    thresholds = {
        "mean radius": 14.0,
        "mean area": 650.0,
        "mean perimeter": 90.0,
        "area error": 40.0,
        "worst radius": 17.0,
        "mean concavity": 0.08,
        "mean concave points": 0.05,
        "worst perimeter": 110.0,
        "worst area": 1000.0,
        "worst concave points": 0.15
    }

    # Grid Layout: 5 columns per row for 10 features
    viz_cols = st.columns(5)
    features = list(thresholds.keys())

    for idx, feature in enumerate(features):
        with viz_cols[idx % 5]:
            val = st.session_state.input_df[feature].values[0]
            thresh = thresholds[feature]
            
            fig, ax = plt.subplots(figsize=(3, 3))
            
            # Background Zones
            ax.axvspan(0, thresh, color='green', alpha=0.1)
            ax.axvspan(thresh, thresh*3, color='red', alpha=0.1)
            ax.axvline(thresh, color='black', linestyle='--', linewidth=0.8)
            
            # Patient Marker
            marker_color = '#22c55e' if val <= thresh else '#ef4444'
            ax.scatter(val, 0.5, color=marker_color, s=120, edgecolors='white', zorder=5)
            
            # Formatting
            ax.set_title(f"{feature.title()}", fontsize=10, fontweight='bold', color='#e2e8f0')
            ax.set_yticks([])
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(val, thresh * 1.6))
            ax.tick_params(axis='x', labelsize=8, colors='#94a3b8')
            
            # Text Status
            status = "Normal" if val <= thresh else "High"
            ax.text(val, 0.25, status, color=marker_color, ha='center', fontsize=9, fontweight='bold')
            
            # Transparent Background for plots
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            for spine in ax.spines.values():
                spine.set_color('#334155')

            st.pyplot(fig)
            plt.close(fig)

    # ---------------- CRITICAL FEATURE RISK ANALYSIS ----------------
    st.markdown("### 🔍 Important Breast Cancer Input Features")
    
    # Define the 5 key features you specified
    critical_features = {
        "mean radius": 14.0,
        "mean perimeter": 90.0,
        "mean concave points": 0.05,
        "worst area": 1000.0,
        "worst concave points": 0.15
    }

    # Create 5 columns for the critical feature cards
    crit_cols = st.columns(5)
    
    all_crossed = True  # Flag to check if ALL values are in the danger zone

    for i, (feature, thresh) in enumerate(critical_features.items()):
        val = st.session_state.input_df[feature].values[0]
        is_danger = val > thresh
        
        if not is_danger:
            all_crossed = False
            
        with crit_cols[i]:
            # Status Indicator
            status_label = "🚨 DANGER" if is_danger else "✅ SAFE"
            status_color = "#ef4444" if is_danger else "#22c55e"
            
            st.markdown(f"""
                <div style="background-color: #1e293b; padding: 10px; border-radius: 10px; border-left: 5px solid {status_color}; text-align: center;">
                    <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">{feature.upper()}</p>
                    <h3 style="margin: 5px 0; color: {status_color};">{status_label}</h3>
                    <p style="margin: 0; font-weight: bold;">{val:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
f_col1, f_col2 = st.columns([4, 1])

with f_col1:
    st.info("**Model Info:** Logistic Regression trained on Wisconsin Breast Cancer Dataset. Features include size, shape, and texture metrics. For educational use only.")

with f_col2:
    if st.button("🔄 Reset Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()