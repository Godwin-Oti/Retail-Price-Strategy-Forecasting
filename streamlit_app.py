# dashboard_app.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib  # For loading your trained model
import numpy as np

# Load data & model
df = pd.read_csv("your_cleaned_data.csv")  # Replace with actual data path
model = joblib.load("your_trained_model.pkl")  # Replace with actual model

# Title
st.title("📊 Sales Prediction Dashboard")

# Sidebar - User inputs for simulation
st.sidebar.header("🔧 Simulation Controls")
unit_price = st.sidebar.slider("Unit Price", float(df.unit_price.min()), float(df.unit_price.max()), float(df.unit_price.mean()))
freight_price = st.sidebar.slider("Freight Price", float(df.freight_price.min()), float(df.freight_price.max()), float(df.freight_price.mean()))
promotion = st.sidebar.selectbox("Promotion", [0, 1])
holiday = st.sidebar.selectbox("Holiday", [0, 1])

# Prediction input
input_data = df.iloc[-1:].copy()
input_data["unit_price"] = unit_price
input_data["freight_price"] = freight_price
input_data["Promotion"] = promotion
input_data["Holiday"] = holiday

# Prediction
prediction = model.predict(input_data.drop(columns=["qty"]))[0]
st.metric("📈 Predicted Quantity", f"{prediction:.2f}")

# SHAP
st.subheader("🔍 SHAP Feature Impact")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data.drop(columns=["qty"]))

# Plot SHAP values
fig, ax = plt.subplots()
shap.summary_plot(shap_values, input_data.drop(columns=["qty"]), plot_type="bar", show=False)
st.pyplot(fig)
