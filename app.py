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

 

st.title("Taxi Fare Prediction App 🚕💰") 

st.write("Enter the trip details below to estimate the taxi fare.") 

 

# Create input fields based on your dataset 

trip_distance = st.number_input("Trip Distance (km)", min_value=0.1, value=2.0) 

passenger_count = st.number_input("Passenger Count", min_value=1, value=1) 

trip_duration = st.number_input("Trip Duration (minutes)", min_value=1.0, value=10.0) 

 

# Prepare input for model 

input_data = pd.DataFrame({ 

    "trip_distance": [trip_distance], 

    "passenger_count": [passenger_count], 

    "trip_duration": [trip_duration] 

}) 

 

# Scale 

input_scaled = scaler.transform(input_data) 

 

# Predict using both models 

rf_pred = rf.predict(input_scaled)[0] 

ada_pred = ada.predict(input_scaled)[0] 

 

# ===================== 

# Show Results 

# ===================== 

 

if st.button("Predict Fare"): 

    st.success(f"Random Forest Predicted Fare: ${rf_pred:.2f}") 

    st.info(f"AdaBoost Predicted Fare: ${ada_pred:.2f}") 

 

    st.write(""" 

    ### Model Performance: 

    - Random Forest provides more accurate and stable predictions. 

    - AdaBoost is more sensitive to noise but still useful for comparison. 

    """) 
