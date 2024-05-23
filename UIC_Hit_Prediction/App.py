import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
import json

# Load models and scaler
models_dir = '/Users/purva/Documents/IDS560_Capstone/Code/Models' 

batter_names_uic = ['Petersen, Rayth', 'Nagelbach, Ryan', 'Conn, Cole', 'Szykowny, Charlie', 'Farfan, Max', 'Nowik, Breck', 'Henkle, AJ', 
                    'Zielinski, Zane', 'Grimes, Bobby', 'Roberts, Carson', 'Conn, Clay', 'TAYLOR, AJ', 'Fisher, Jackson', 'Ewell, Kendal', 
                    'Kay, Garrett', 'Heusohn, Marcus', 'Harris, James', 'Colon, Vidol', 'Szuba, Caden', 'Nicoloudes, Pambos', 
                    'Bessette, Jackson', 'Snyder, Jack', 'Kim, Zachary', 'Smith, Lucas', 'Camp, Camp', 'Dee, Sean', 'Zahora, Matt', 
                    'Schueler, Dillon', 'Lyons, Kendall', 'Butler, DJ', 'Bak, Brandon', 'Gohlke, Vincent']

mapping_file = os.path.join(models_dir, 'batter_name_to_id_mapping.json')
with open(mapping_file, 'r') as f:
    name_to_id_uic = json.load(f)

# Function to load a model and its corresponding scaler
def load_model_and_scaler(batter_name):
    batter_id = str(name_to_id_uic.get(batter_name, 'general'))  # Convert to string if necessary
    model_path = os.path.join(models_dir, f'{batter_id}_model.joblib')
    scaler_path = os.path.join(models_dir, f'{batter_id}_scaler.joblib')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler
    return None, None

# Prediction function
def predict_hit_with_probability(batter_name, RelSpeed, SpinRate, PlateLocHeight, PlateLocSide):
    model, scaler = load_model_and_scaler(batter_name)
    if model is not None and scaler is not None:
        features = np.array([[RelSpeed, SpinRate, PlateLocHeight, PlateLocSide]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0, prediction]
        return f"Hit Probability: {probability:.2%}"
    else:
        return "Model not found for the selected batter."

# Streamlit app interface
st.title("Hit Prediction for UIC Baseball Team")

selected_batter = st.selectbox("Select Batter", options=batter_names_uic)

rel_speed = st.slider("Release Speed", min_value=0.0, max_value=100.0, value=50.0)
spin_rate = st.slider("Spin Rate", min_value=0.0, max_value=5000.0, value=2500.0)
# Using columns to create a side-by-side layout for PlateLoc sliders
col1, col2 = st.columns(2)

with col1:
    st.text("Plate Location Side")  # Adding text to mimic a vertical slider label
    plate_loc_side = st.slider("", -10.0, 10.0, 0.0, key='plate_loc_side')

with col2:
    st.text("Plate Location Height")  # Adding text for clarity
    plate_loc_height = st.slider("", 0.0, 5.0, 2.5, key='plate_loc_height')

if st.button("Predict Hit"):
    if selected_batter:
        result = predict_hit_with_probability(selected_batter, rel_speed, spin_rate, plate_loc_height, plate_loc_side)
        st.success(f"Prediction for {selected_batter}: {result}")
    else:
        st.error("Please select a batter from the dropdown.")

