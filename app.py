# app.py
import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import st_folium

# -------------------------
# Load trained ML model
# -------------------------
with open("model/earthquake_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# App title
# -------------------------
st.set_page_config(page_title="Earthquake Prediction", layout="wide")
st.title("🌎 Earthquake Prediction & Visualization App")
st.write("Predict earthquake severity and explore historical earthquakes on an interactive map.")

# -------------------------
# User input for prediction
# -------------------------
st.sidebar.header("Predict Earthquake")
magnitude = st.sidebar.number_input("Magnitude (Richter scale)", min_value=0.0, max_value=10.0, step=0.1)
depth = st.sidebar.number_input("Depth (km)", min_value=0.0, max_value=700.0, step=1.0)

if st.sidebar.button("Predict"):
    prediction = model.predict([[magnitude, depth]])
    st.sidebar.success(f"Predicted Earthquake Severity: {prediction[0]}")

# -------------------------
# Visualize earthquakes
# -------------------------
st.header("🌐 Historical Earthquakes Map")
st.write("Explore past earthquakes around the world.")

# Load historical earthquake data
# Ensure you have a CSV with columns: latitude, longitude, magnitude
try:
    earthquake_data = pd.read_csv("data/earthquakes.csv")
except FileNotFoundError:
    st.warning("Historical earthquake data not found. Map will be empty.")
    earthquake_data = pd.DataFrame(columns=["latitude", "longitude", "magnitude"])

# Initialize map
m = folium.Map(location=[0, 0], zoom_start=2)

# Add markers
for _, row in earthquake_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['magnitude'] * 2,  # scale marker size by magnitude
        color='red',
        fill=True,
        fill_opacity=0.6,
        popup=f"Magnitude: {row['magnitude']}"
    ).add_to(m)

# Display map
st_data = st_folium(m, width=800, height=500)

# -------------------------
# Optional: Display raw data
# -------------------------
with st.expander("View Raw Earthquake Data"):
    st.dataframe(earthquake_data)
    st.download_button(
        label="Download data as CSV",
        data=earthquake_data.to_csv(index=False),
        file_name='earthquakes.csv',
        mime='text/csv',
    )
# -------------------------
st.markdown("---")
st.markdown("Developed by Palaksh kumar")   
st.markdown("© 2024 Earthquake Prediction App")
st.markdown("Data Source: [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/)")
st.markdown("Model: Simple RandomForestClassifier trained on magnitude and depth")
st.markdown("Visualization: Folium for interactive maps")
st.markdown("Built with Streamlit")
st.markdown("Feel free to explore and contribute!")
st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/palaksh-kumar-4b3b021b4/) | [GitHub](https://github.com/Palaksh-singh)")
st.markdown("Thank you for using the Earthquake Prediction & Visualization App! 🌍")