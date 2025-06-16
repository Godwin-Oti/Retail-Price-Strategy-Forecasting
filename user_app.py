import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

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

        # --- SHAP Explanation with Plotly ---
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

        # Get SHAP values and feature names
        preprocessed_input = model.named_steps['preprocessor'].transform(input_data)
        feature_names = get_feature_names(model)

        explainer = shap.Explainer(model.named_steps['model'], feature_names=feature_names)
        shap_values = explainer(preprocessed_input)

        # Build dataframe for Plotly
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0].values
        }).sort_values(by="SHAP Value", key=abs, ascending=True)

        fig_shap = go.Figure(
            go.Bar(
                x=shap_df["SHAP Value"],
                y=shap_df["Feature"],
                orientation='h',
                marker=dict(color=shap_df["SHAP Value"], colorscale="RdBu"),
            )
        )
        fig_shap.update_layout(
            title="Feature Contribution to Prediction",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # --- Display Raw Input Used ---
        st.subheader("🔎 Model Input")
        st.write(input_data.reset_index(drop=True))

        # --- Historical Sales Trend with Plotly ---
        st.subheader("📉 Historical Sales Trend")

        history_df = df[df["product_id"] == product_id].copy()
        history_df["month_year"] = pd.to_datetime(
            history_df["year"].astype(str) + "-" + history_df["month"].astype(str),
            format="%Y-%m"
        )
        history_df = history_df.sort_values("month_year")

        fig_trend = px.line(
            history_df,
            x="month_year",
            y="total_quantity_sold",
            markers=True,
            title=f"Sales Trend for Product ID: {product_id}",
            labels={"total_quantity_sold": "Quantity Sold", "month_year": "Month"}
        )
        selected_date = pd.to_datetime(f"{input_data['year'].values[0]}-{input_data['month'].values[0]}")
        fig_trend.add_vline(x=selected_date, line_dash="dash", line_color="red")

        st.plotly_chart(fig_trend, use_container_width=True)
