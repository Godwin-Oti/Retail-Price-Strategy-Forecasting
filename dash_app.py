# dashboard_app.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# --- Load Data & Model ---
df = pd.read_csv("processed_retail_data.csv")
model = joblib.load("sales_pipeline.pkl")  # Your trained model

# --- Title ---
st.title("📊 Sales Quantity Prediction Dashboard")

# --- Sidebar: Simulation Controls ---
st.sidebar.header("🔧 Simulation Controls")

unit_price = st.sidebar.slider(
    "Unit Price", float(df.unit_price.min()), float(df.unit_price.max()), float(df.unit_price.mean())
)
freight_price = st.sidebar.slider(
    "Freight Price", float(df.freight_price.min()), float(df.freight_price.max()), float(df.freight_price.mean())
)
promotion = st.sidebar.selectbox("Promotion", [0, 1])
holiday = st.sidebar.selectbox("Holiday", [0, 1])

# --- Prepare Prediction Input ---
input_data = df.iloc[[-1]].copy()  # Last row as template
input_data["unit_price"] = unit_price
input_data["freight_price"] = freight_price

# Match column names (adjust if lowercase is used)
for col in ["promotion", "Promotion"]:
    if col in input_data.columns:
        input_data[col] = promotion
for col in ["holiday", "Holiday"]:
    if col in input_data.columns:
        input_data[col] = holiday

# --- Prediction ---
X_input = input_data.drop(columns=["qty"], errors='ignore')
prediction = model.predict(X_input)[0]
st.metric("📦 Predicted Quantity Sold", f"{prediction:.2f}")

# --- SHAP Feature Impact ---
st.subheader("🔍 Feature Importance (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(X_input)

# Plot SHAP summary bar chart
fig, ax = plt.subplots()
shap.plots.bar(shap_values[0], show=False)
st.pyplot(fig)
