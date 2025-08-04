import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

# Simulated data
data = {
    'steering_angle': [0, 2, -18, 20, -25, 0, 15, -20, 5, 0],
    'speed': [30, 28, 12, 10, 8, 35, 18, 14, 32, 33],
    'yaw_rate': [0.1, 0.2, -0.5, 0.6, -0.8, 0.0, 0.3, -0.7, 0.1, 0.0],
    'turn_intent': [0, 0, -1, 1, -1, 0, 1, -1, 0, 0]
}
df = pd.DataFrame(data)

# Model training
X = df[['steering_angle', 'speed', 'yaw_rate']]
y = df['turn_intent']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['steering_angle', 'yaw_rate']])
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_scaled, y)

# Streamlit UI
st.title("🚕 AutoPilot – AI-Powered Turn Intent Prediction")
st.markdown("This app predicts whether an auto is about to turn left, right, or go straight based on motion inputs. All simulated for fun (for now 😄).")

# Input sliders
steering = st.slider("Steering Angle (°)", -30, 30, 0)
speed = st.slider("Speed (km/h)", 0, 50, 25)
yaw = st.slider("Yaw Rate (simulated)", -1.0, 1.0, 0.0)

# Prediction
input_df = pd.DataFrame([[steering, speed, yaw]], columns=['steering_angle', 'speed', 'yaw_rate'])
scaled_input = scaler.transform(input_df[['steering_angle', 'yaw_rate']])
prediction = model.predict(scaled_input)[0]

if prediction == 1:
    st.success("🔁 Turn Right Detected – Indicator ON")
elif prediction == -1:
    st.warning("🔄 Turn Left Detected – Indicator ON")
else:
    st.info("⬆️ Going Straight – Indicator OFF")

# Show decision boundary
st.subheader("🧠 Model Decision Boundary")
st.image("decision_boundary_plot.png", caption="Decision zones based on steering & yaw")

# Show simulated auto view
st.subheader("🚗 Simulated Auto View (Back Camera)")
st.image("auto_view_frame.png", caption="Auto in front with coordinate overlay")
