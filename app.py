import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model & data
model = joblib.load("pdm_model.pkl")
df = pd.read_csv("sensor_data.csv")

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("⚙️ Predictive Maintenance Dashboard")
st.write("Monitor machine health and predict failures using AI")

# Sidebar for user input
st.sidebar.header("🔍 Enter Sensor Readings")
temp = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=200.0, step=0.1)
vib = st.sidebar.number_input("Vibration (mm/s)", min_value=0.0, max_value=50.0, step=0.1)

if st.sidebar.button("Predict"):
    result = model.predict([[temp, vib]])[0]
    if result == 0:
        st.success(f"✅ Machine is Healthy (Temp={temp}, Vib={vib})")
    else:
        st.error(f"⚠️ Machine at Risk of Failure (Temp={temp}, Vib={vib})")

st.markdown("---")

# Show charts
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(df, x="time", y="temperature", title="Temperature Over Time", markers=True)
    fig1.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Failure Threshold")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(df, x="time", y="vibration", title="Vibration Over Time", markers=True)
    fig2.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="Failure Threshold")
    st.plotly_chart(fig2, use_container_width=True)

