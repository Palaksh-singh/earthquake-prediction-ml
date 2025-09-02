import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import tensorflow as tf
import numpy as np

# Page settings
st.set_page_config(page_title="Earthquake Prediction", page_icon="🌍", layout="wide")

# Sidebar
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Predict", "About"])

# Load model (once)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("models/earth_model.h5")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        return None

model = load_model()

# Home page
if page == "Home":
    st.title("🌍 Earthquake Prediction Dashboard")
    st.markdown(
        """
        Welcome to the **Earthquake Prediction ML App**.  
        This tool uses **Machine Learning** to analyze earthquake data and make predictions.

        **Features:**
        - Upload earthquake dataset (CSV)  
        - Predict **magnitude** & **depth**  
        - Visualize earthquakes on an **interactive world map**  
        - Get performance metrics  

        ---
        """
    )

# Upload & Predict page
elif page == "Upload & Predict":
    st.title("📂 Upload Dataset & Predict")

    uploaded_file = st.file_uploader("Upload your earthquake dataset (CSV)", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### 📊 Dataset Preview", data.head())

        # Ensure required columns
        if {"latitude", "longitude", "depth", "magnitude"}.issubset(data.columns):
            # Prepare features
            X = data[["latitude", "longitude", "depth"]].values

            if model:
                # Predictions
                preds = model.predict(X)
                data["predicted_magnitude"] = preds[:, 0]
                data["predicted_depth"] = preds[:, 1]

                # Metrics section
                st.subheader("📈 Model Performance (on uploaded data)")
                col1, col2 = st.columns(2)

                mae_mag = np.mean(np.abs(data["magnitude"] - data["predicted_magnitude"]))
                mae_depth = np.mean(np.abs(data["depth"] - data["predicted_depth"]))

                col1.metric("MAE Magnitude", f"{mae_mag:.3f}")
                col2.metric("MAE Depth", f"{mae_depth:.2f}")

                # Tabs for results
                tab1, tab2 = st.tabs(["🔍 Predictions", "🗺️ Map View"])

                with tab1:
                    st.write("### Predictions (first 10 rows)")
                    st.dataframe(data.head(10))

                with tab2:
                    st.write("### Earthquake Map")
                    m = folium.Map(location=[20, 0], zoom_start=2)

                    for _, row in data.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=3,
                            popup=f"True Mag: {row['magnitude']}, Pred Mag: {row['predicted_magnitude']:.2f}, Depth: {row['depth']}",
                            color="red",
                            fill=True
                        ).add_to(m)

                    st_folium(m, width=800, height=500)
        else:
            st.error("CSV must include: `latitude`, `longitude`, `depth`, `magnitude`")

# About page
elif page == "About":
    st.title("ℹ️ About this Project")
    st.markdown(
        """
        This project demonstrates **Machine Learning for Earthquake Prediction**.  
        Built using:
        - 🐍 Python, Pandas, TensorFlow  
        - 🌍 Folium for interactive maps  
        - 🎨 Streamlit for web app  

        **Developer:** Palaksh Kumar  
        📧 [Email](mailto:palakshkumar866@gmail.com)  
        💻 [GitHub](https://github.com/Palaksh-singh)  
        🔗 [LinkedIn](https://www.linkedin.com/in/palaksh-kumar-584674346/)  
        📸 [Instagram](https://www.instagram.com/palakshkumar_)  
        """
    )
