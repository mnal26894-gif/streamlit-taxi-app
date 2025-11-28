import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================
# Load Models
# =====================

rf = joblib.load("random_forest_model.pkl")
ada = joblib.load("adaboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# =====================
# Streamlit UI
# =====================

st.set_page_config(page_title="Taxi Fare Prediction", page_icon="🚕", layout="centered")

st.title("🚕 Taxi Fare Prediction System")
st.write("Enter the trip details below to estimate the taxi fare using Machine Learning.")

# =====================
# Input Fields (Matching the Report)
# =====================

trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, value=5.0)
trip_duration = st.number_input("Trip Duration (minutes)", min_value=1.0, value=15.0)
base_fare = st.number_input("Base Fare ($)", min_value=0.0, value=3.0)
per_km_rate = st.number_input("Per Kilometer Rate ($)", min_value=0.0, value=1.5)
per_minute_rate = st.number_input("Per Minute Rate ($)", min_value=0.0, value=0.5)
passenger_count = st.number_input("Passenger Count", min_value=1, value=1)

# =====================
# Prepare Input for Model (Exact Feature Order)
# =====================

input_array = np.array([[
    trip_distance,
    trip_duration,
    base_fare,
    per_km_rate,
    per_minute_rate,
    passenger_count
]])

input_scaled = scaler.transform(input_array)

# =====================
# Feature Scaling
# =====================

input_scaled = scaler.transform(input_array)

# =====================
# Predictions
# =====================

rf_pred = rf.predict(input_scaled)[0]
ada_pred = ada.predict(input_scaled)[0]

# =====================
# Show Results
# =====================

if st.button("💰 Predict Fare"):
    st.success(f"✅ Random Forest Predicted Fare: ${rf_pred:.2f}")
    st.info(f"ℹ️ AdaBoost Predicted Fare: ${ada_pred:.2f}")

    st.markdown("""
    ### ✅ Model Summary:
    - Random Forest provides higher accuracy and stability.
    - AdaBoost is more sensitive to noise but useful for model comparison.
    """)

st.markdown("---")
st.caption("Taxi Fare Prediction System | Applied Machine Learning Mini Project")
