import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --- Load Data & Model ---
df = pd.read_csv("processed_retail_data.csv")
model = joblib.load("sales_pipeline.pkl")

# --- Title ---
st.title("📦 Sales Quantity Prediction Dashboard")

# --- Sidebar: Product & Month Selection ---
st.sidebar.header("🔍 Select Product & Month")

product_id = st.sidebar.selectbox("Product ID", df['product_id'].unique())
available_months = df[df['product_id'] == product_id]['month'].dropna().unique()

if len(available_months) == 0:
    st.warning("No months available for the selected product. Please choose another product.")
    st.stop()

selected_month = st.sidebar.selectbox("Month", sorted(available_months))

# --- Sidebar: Simulation Controls ---
st.sidebar.header("⚙️ Adjust Simulation Inputs")

# Confirm column existence before accessing
unit_price_col = 'unit_price' if 'unit_price' in df.columns else None
freight_price_col = 'freight_price' if 'freight_price' in df.columns else None

if not unit_price_col or not freight_price_col:
    st.error("Missing required columns in the dataset: 'unit_price' or 'freight_price'. Please check your input data.")
    st.stop()

unit_price = st.sidebar.slider(
    "Unit Price", float(df[unit_price_col].min()), float(df[unit_price_col].max()), float(df[unit_price_col].mean())
)
freight_price = st.sidebar.slider(
    "Freight Price", float(df[freight_price_col].min()), float(df[freight_price_col].max()), float(df[freight_price_col].mean())
)
holiday_flag = st.sidebar.selectbox("Holiday", [0, 1])

# --- Prepare Input for Prediction ---
lookup_df = df[(df['product_id'] == product_id) & (df['month'] == selected_month)]

if lookup_df.empty:
    st.warning("No data available for the selected product and month. Try another combination.")
else:
    required_cols = ['product_category_name', 'month', 'year', 'month_index', 'lag_1', 'unit_price', 'freight_price', 'holiday']
    missing_cols = [col for col in required_cols if col not in lookup_df.columns]

    if missing_cols:
        st.error(f"The following required columns are missing in your dataset: {', '.join(missing_cols)}")
    else:
        input_data = lookup_df[required_cols].copy()

        # --- Apply Simulated Input ---
        input_data['unit_price'] = unit_price
        input_data['freight_price'] = freight_price
        input_data['holiday'] = holiday_flag

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

                if isinstance(transformer, Pipeline):
                    last_step = transformer.steps[-1][1]
                    if hasattr(last_step, "get_feature_names_out"):
                        try:
                            names = last_step.get_feature_names_out(cols)
                        except:
                            names = cols
                    else:
                        names = cols
                elif hasattr(transformer, "get_feature_names_out"):
                    try:
                        names = transformer.get_feature_names_out(cols)
                    except:
                        names = cols
                else:
                    names = cols

                cleaned_names = [name.replace("num__", "").replace("cat__", "") for name in names]
                output_features.extend(cleaned_names)

            return output_features

        # Get preprocessed input and feature names
        preprocessed_input = model.named_steps['preprocessor'].transform(input_data)
        feature_names = get_feature_names(model)

        explainer = shap.Explainer(model.named_steps['model'], feature_names=feature_names)
        shap_values = explainer(preprocessed_input)

        # Plot SHAP with clean feature names
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        # --- Display Raw Input Used ---
        st.subheader("🔎 Model Input")
        st.write(input_data.reset_index(drop=True))
