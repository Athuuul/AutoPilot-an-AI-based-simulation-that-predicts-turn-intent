# AutoPilot – Predictive Turn Indicator AI for Auto‑Rickshaws

A prototype ML system that predicts turn intent (left/right/straight) using simulated motion data — built with Python and Streamlit, with live demo and visualization.

## Key Features
- Simulated inputs: steering angle, speed, yaw rate
- Predicts turn intent using Logistic Regression
- Real-time Streamlit UI
- Decision boundary and mock auto-view visuals

## How to Run
```bash
git clone https://github.com/Athuuuul/AutoPilot-an-AI-based-simulation-that-predicts-turn-intent.git
cd AutoPilot-an-AI-based-simulation-that-predicts-turn-intent
pip install -r requirements.txt
streamlit run app.py
