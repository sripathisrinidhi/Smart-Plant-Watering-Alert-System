import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
st.set_page_config(
    page_title="Smart Plant Care System",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Define Constants and Mappings ---
STATE_MAP = {0: 'Healthy', 1: 'Needs Water', 2: 'Overwatered'}
MODEL_PATH = 'random_forest_model.joblib'
SCALER_PATH = 'scaler.joblib'

# --- Utility Functions ---

# Use Streamlit's caching mechanism to load the large files once
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and scaler using joblib."""
    try:
        # Check if the files exist before loading (CRITICAL for deployment)
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error(f"Missing required files: {MODEL_PATH} and/or {SCALER_PATH}")
            st.warning("Please ensure you ran the 'train_and_save_model.py' script locally and uploaded the resulting .joblib files to your GitHub repository.")
            return None, None
            
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model assets. Check if the files are correct: {e}")
        return None, None


def get_plant_alert(temp, hum, light, soil_moisture, model, scaler):
    """Predicts the plant state and generates an alert."""
    
    # 1. Create a DataFrame for the new input (Must match the training features order!)
    new_data = pd.DataFrame({
        'Temperature': [temp],
        'Humidity': [hum],
        'Light': [light],
        'SoilMoisture': [soil_moisture]
    })

    # 2. Scale the input data
    new_data_scaled = scaler.transform(new_data)

    # 3. Predict the state and probabilities
    prediction_label = model.predict(new_data_scaled)[0]
    prediction_state = STATE_MAP[prediction_label]

    probabilities = model.predict_proba(new_data_scaled)[0]
    confidence_score = probabilities[prediction_label]
    all_probabilities = {STATE_MAP[i]: probabilities[i] for i in range(len(STATE_MAP))}

    # 4. Generate the alert and recommendation
    recommendation = ""
    
    if prediction_state == 'Needs Water':
        alert = "💧 ACTION REQUIRED: Soil is too dry!"
        recommendation = "Recommendation: Water your plant immediately. Low soil moisture is stressing the plant."
    elif prediction_state == 'Overwatered':
        alert = "⚠️ WARNING: Soil is too wet!"
        recommendation = "Recommendation: Stop watering for several days. Excessive moisture can lead to root rot."
    else:
        alert = "✅ STATUS: Optimal conditions detected."
        recommendation = "Recommendation: Your plant is happy and healthy! Maintain the current sensor levels."

    return {
        'State': prediction_state,
        'Confidence': confidence_score,
        'Alert': alert,
        'Recommendation': recommendation,
        'All_Probabilities': all_probabilities
    }

# --- Main Streamlit Application UI ---

st.title("🌿 Smart Plant Health Classifier")
st.markdown("Use sensor readings to predict your plant's current state based on a trained Random Forest model.")

# Load the model and scaler
model, scaler = load_assets()

if model is None or scaler is None:
    st.stop() # Stop execution if assets are missing

st.header("1. Input Sensor Readings", divider='green')

with st.form("plant_input_form"):
    
    # Input sliders for the four features
    col_temp, col_hum = st.columns(2)
    with col_temp:
        temp = st.slider('Temperature (°C)', min_value=15.0, max_value=35.0, value=23.0, step=0.1, help="Optimal range for simulation: 20-26°C.")
    with col_hum:
        hum = st.slider('Humidity (%)', min_value=20.0, max_value=90.0, value=55.0, step=0.1, help="Optimal range for simulation: 40-60%.")
    
    col_light, col_soil = st.columns(2)
    with col_light:
        light = st.slider('Light (Lux)', min_value=100.0, max_value=1000.0, value=650.0, step=1.0, help="Optimal range for simulation: 500-800 Lux.")
    with col_soil:
        soil_moisture = st.slider('Soil Moisture (%)', min_value=0.0, max_value=100.0, value=50.0, step=0.1, help="Optimal range for simulation: 40-65%.")
    
    submit_button = st.form_submit_button("Get Plant Diagnosis")

# --- Display Results ---

if submit_button:
    st.header("2. Plant Health Diagnosis", divider='orange')
    
    with st.spinner("Analyzing sensor data..."):
        
        results = get_plant_alert(temp, hum, light, soil_moisture, model, scaler)
        
        # Display alert box
        if results['State'] == 'Healthy':
            st.success(f"**{results['Alert']}**")
        elif results['State'] == 'Needs Water':
            st.warning(f"**{results['Alert']}**")
        else:
            st.error(f"**{results['Alert']}**")

        # Display details
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted State", value=results['State'], delta=None)
            
        with col2:
            st.metric(label="Confidence Score", value=f"{results['Confidence'] * 100:.2f}%", delta=None)
        
        # Display Recommendation
        st.subheader("Action Plan")
        st.info(results['Recommendation'])

        # Display Probability Chart
        st.subheader("Model Certainty")
        
        chart_data = pd.DataFrame(
            {'State': list(results['All_Probabilities'].keys()), 
             'Probability': list(results['All_Probabilities'].values())}
        ).sort_values(by='Probability', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='State', y='Probability', data=chart_data, palette='viridis', ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title('Probability Distribution of Plant States')
        ax.set_ylabel('Probability')
        ax.set_xlabel('Plant State')
        st.pyplot(fig)
