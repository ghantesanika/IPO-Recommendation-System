import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="IPO Recommendation System", layout="centered")

st.title("📊 IPO Recommendation & Allotment Probability System")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    reco_model = joblib.load("models/xgb_smote_safe.joblib")
    label_encoder = joblib.load("models/label_encoder_smote_safe.joblib")

    allot_bundle = joblib.load("models/allotment_model.joblib")
    allot_model = allot_bundle["model"]
    allot_scaler = allot_bundle["scaler"]
    allot_features = allot_bundle["features"]

    return reco_model, label_encoder, allot_model, allot_scaler, allot_features


reco_model, label_encoder, allot_model, allot_scaler, allot_features = load_models()

# ---------------- USER INPUT ----------------
st.subheader("Enter IPO Details")

issue_price = st.number_input("Issue Price (Rs)", min_value=0.0, value=100.0)
ipo_duration = st.number_input("IPO Duration (Days)", min_value=1, value=3)
#gmp = st.number_input("Grey Market Premium (GMP)", value=20.0)
log_issue_size = st.number_input("Log Issue Size", value=6.5)
sub_strength = st.number_input("Subscription Strength", value=10.0)

retail_sub = st.number_input("Retail Subscription", value=5.0)
qib_sub = st.number_input("QIB Subscription", value=10.0)
nii_sub = st.number_input("NII Subscription", value=8.0)
price_band_width = st.number_input("Price Band Width", value=10.0)

# ---------------- PREDICT ----------------
if st.button("🔍 Predict"):

    # ----- Recommendation model input -----
    X_reco = pd.DataFrame([{
        "Issue Price (Rs)": issue_price,
        "IPO_Duration": ipo_duration,
        "log_Issue_Size": log_issue_size,
        "Sub_Strength": sub_strength
    }])

    reco_encoded = reco_model.predict(X_reco)[0]
    reco_label = label_encoder.inverse_transform([reco_encoded])[0]

    st.success(f"📌 Recommendation: **{reco_label}**")

    # ----- Allotment probability model input -----
    X_allot = pd.DataFrame([{
        "Retail Subscription": retail_sub,
        "Subscription QIB": qib_sub,
        "Subscription NII": nii_sub,
        "log_Issue_Size": log_issue_size,
        "Price_Band_Width": price_band_width,
        "Sub_Strength": sub_strength
    }])

    # Ensure correct feature order
    X_allot = X_allot[allot_features]

    X_allot_scaled = allot_scaler.transform(X_allot)
    allot_prob = allot_model.predict(X_allot_scaled)[0]

    st.info(f"🎯 Estimated Allotment Probability: **{round(allot_prob * 100, 2)}%**")
