# app.py
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🌍 Earthquake Prediction ML App", layout="wide")
st.title("🌍 Earthquake Prediction & Visualization App")

# ==============================
# LOAD MODEL & SCALER
# ==============================
@st.cache_resource
def load_model_and_scaler():
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model("models/earth_model.h5")
    except Exception as e:
        st.warning(f"⚠️ Model not found or failed to load: {e}")
    try:
        scaler = joblib.load("models/scaler.save")
    except Exception as e:
        st.warning(f"⚠️ Scaler not found or failed to load: {e}")
    return model, scaler

model, scaler = load_model_and_scaler()

# ==============================
# SIDEBAR NAVIGATION
# ==============================
page = st.sidebar.radio("📌 Navigation", ["Home", "Model Prediction", "Visualize Earthquakes"])

# ==============================
# HOME PAGE
# ==============================
if page == "Home":
    st.header("📖 Project Overview")
    st.write("""
    Welcome to the **Earthquake Prediction & Visualization App**!  
    This app uses **Machine Learning** to predict earthquake **magnitude and depth** based on seismic data, 
    and provides an **interactive map** for exploring historical earthquake events worldwide.
    """)
    st.markdown("""
    **Features:**  
    - Predict earthquake magnitude & depth from latitude, longitude, and depth input  
    - Interactive map of historical earthquakes with magnitude-scaled markers  
    - Easy navigation using sidebar  
    """)

# ==============================
# MODEL PREDICTION PAGE
# ==============================
elif page == "Model Prediction":
    st.header("🔮 Predict Earthquake Parameters")
    st.write("Enter the earthquake coordinates and depth to get a prediction:")

    # Input fields
    lat = st.number_input("Latitude", value=20.0, format="%.6f")
    lon = st.number_input("Longitude", value=80.0, format="%.6f")
    depth_input = st.number_input("Depth (km)", value=10.0, format="%.2f")

    if st.button("Predict"):
        if model and scaler:
            try:
                X_input = np.array([[lat, lon, depth_input]])
                X_scaled = scaler.transform(X_input)
                
                with st.spinner("Predicting..."):
                    prediction = model.predict(X_scaled)
                    pred_mag, pred_depth = prediction[0]

                st.success(f"✅ Predicted Magnitude: {pred_mag:.2f}")
                st.info(f"🌊 Predicted Depth: {pred_depth:.2f} km")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        else:
            st.warning("⚠️ Model or scaler not available. Please check the files in `models/`.")

# ==============================
# VISUALIZE EARTHQUAKES PAGE
# ==============================
elif page == "Visualize Earthquakes":
    st.header("🗺️ Historical Earthquake Map")
    st.write("Explore earthquake events worldwide. Marker size scales with magnitude.")

    # Load earthquake data
    try:
        data = pd.read_csv("data/processed.csv")  # Ensure columns: latitude, longitude, magnitude, depth
    except Exception as e:
        st.error(f"⚠️ Could not load earthquake data: {e}")
        data = pd.DataFrame(columns=["latitude", "longitude", "magnitude", "depth"])

    # Filter data (optional)
    min_mag, max_mag = st.slider("Filter by Magnitude", 0.0, 10.0, (0.0, 10.0), 0.1)
    filtered_data = data[(data["magnitude"] >= min_mag) & (data["magnitude"] <= max_mag)]

    # Initialize map
    m = folium.Map(location=[20, 0], zoom_start=2)
    for _, row in filtered_data.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(2, row["magnitude"]*2),  # scale radius by magnitude
            popup=f"Magnitude: {row['magnitude']:.2f}, Depth: {row['depth']:.2f} km",
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.6
        ).add_to(m)

    # Embed map
    st_folium(m, width=900, height=600)

    # Optional: Display raw data
    with st.expander("View Raw Earthquake Data"):
        st.dataframe(filtered_data)
