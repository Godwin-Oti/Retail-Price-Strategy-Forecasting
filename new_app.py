import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load Data & Model ---
df = pd.read_csv("processed_retail_data.csv")
model = joblib.load("sales_pipeline.pkl")

# --- Title ---
st.title("📦 Sales Quantity Prediction Dashboard")

# --- Sidebar: Product & Month Selection ---
st.sidebar.header("🔍 Select Product & Month")

product_id = st.sidebar.selectbox("Product ID", sorted(df['product_id'].unique()))

# Make sure 'month' is integer type
df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')

# Filter available months for selected product
product_data = df[df['product_id'] == product_id]
available_months = sorted(product_data['month'].dropna().unique())

if not available_months:
    st.warning("No months available for the selected product.")
    st.stop()

selected_month = st.sidebar.selectbox("Month", available_months)

# --- Prepare Input for Prediction ---
lookup_df = df[(df['product_id'] == product_id) & (df['month'] == selected_month)]

if lookup_df.empty:
    st.warning("No data available for the selected product and month. Try another combination.")
else:
    input_data = lookup_df[['product_category_name', 'month', 'year', 'month_index', 'lag_1']]

    # --- Prediction ---
    prediction = model.predict(input_data)[0]
    st.metric("📈 Predicted Quantity Sold", f"{prediction:.2f}")

    # --- SHAP Explanation ---
    st.subheader("🔍 Feature Impact (SHAP Values)")

    def get_feature_names(pipeline):
        preprocessor = pipeline.named_steps["preprocessor"]
        output_features = []

        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue

            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(cols)
            else:
                names = cols

            output_features.extend(names)

        return output_features

    # Get preprocessed input and feature names
    preprocessed_input = model.named_steps['preprocessor'].transform(input_data)
    feature_names = get_feature_names(model)

    explainer = shap.Explainer(model.named_steps['model'], feature_names=feature_names)
    shap_values = explainer(preprocessed_input)

    # Plot SHAP with real feature names
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)

    # --- Display Raw Input Used ---
    st.subheader("🔎 Model Input")
    st.write(input_data.reset_index(drop=True))
