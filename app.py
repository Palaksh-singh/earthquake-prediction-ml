import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import tensorflow as tf

st.set_page_config(page_title="Earthquake Prediction", layout="wide")

st.title("🌍 Earthquake Prediction with Machine Learning")

# Upload CSV
uploaded_file = st.file_uploader("Upload your earthquake dataset (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # Load model
    try:
        model = tf.keras.models.load_model("models/earth_model.h5")
        st.success("Model loaded successfully!")

        # Run predictions (example: predict magnitude & depth)
        if {"latitude", "longitude", "depth", "magnitude"}.issubset(data.columns):
            X = data[["latitude", "longitude", "depth"]]
            predictions = model.predict(X)

            st.write("### Predictions (first 5 rows)")
            st.write(predictions[:5])

            # Map
            st.write("### Earthquake Map")
            m = folium.Map(location=[20, 0], zoom_start=2)
            for i, row in data.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    popup=f"Magnitude: {row['magnitude']}, Depth: {row['depth']}",
                    color="red",
                    fill=True
                ).add_to(m)

            st_data = st_folium(m, width=700, height=500)
    except Exception as e:
        st.error(f"Error loading model or processing data: {e}")
