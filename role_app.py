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

# Get unique months for selected product and convert to month names
available_months = sorted(df[df['product_id'] == product_id]['month'].dropna().unique())
month_names = [pd.to_datetime(str(m), format='%m').strftime('%B') for m in available_months]

if not available_months:
    st.warning("No months available for the selected product. Please choose another product.")
    st.stop()

selected_month_name = st.sidebar.selectbox("Month", month_names)
selected_month = available_months[month_names.index(selected_month_name)]

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

# Filter data for selected product and month
lookup_df = df[(df['product_id'] == product_id) & (df['month'] == selected_month)]

if lookup_df.empty:
    st.warning("No data available for the selected product and month. Try another combination.")
    st.stop()

# Pick data from the latest year available for this product-month
latest_year = lookup_df['year'].max()
lookup_df = lookup_df[lookup_df['year'] == latest_year]

required_cols = ['product_category_name', 'month', 'year', 'month_index', 'lag_1', 'unit_price', 'freight_price', 'holiday']
missing_cols = [col for col in required_cols if col not in lookup_df.columns]

if missing_cols:
    st.error(f"The following required columns are missing in your dataset: {', '.join(missing_cols)}")
    st.stop()

input_data = lookup_df[required_cols].copy()

# --- Apply Simulated Input ---
input_data['unit_price'] = unit_price
input_data['freight_price'] = freight_price
input_data['holiday'] = holiday_flag

# --- Prediction ---
prediction = model.predict(input_data)[0]
st.metric("📈 Predicted Quantity Sold", f"{prediction:.2f}")

# --- TABS FOR FUNCTIONAL VIEWS ---
tabs = st.tabs(["Executive", "Sales", "Marketing", "Finance", "Supply Chain", "Model Explanation"])

# --- Executive Summary ---
with tabs[0]:
    st.subheader("📊 Executive Overview")

    this_year = df[df['year'] == df['year'].max()]['total_quantity_sold'].sum()
    last_year = df[df['year'] == df['year'].max() - 1]['total_quantity_sold'].sum()
    growth = ((this_year - last_year) / last_year) * 100 if last_year != 0 else 0

    st.metric("Total Units Sold (YTD)", f"{this_year:,}")
    st.metric("Year-over-Year Growth", f"{growth:.2f}%")

    cat_summary = df.groupby("product_category_name")["total_quantity_sold"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(cat_summary)

    trend_df = df.groupby(['year', 'month'])['total_quantity_sold'].sum().reset_index()
    trend_df['month_year'] = pd.to_datetime(trend_df['year'].astype(str) + '-' + trend_df['month'].astype(str))
    fig_exec_trend = px.line(trend_df, x='month_year', y='total_quantity_sold', title='Overall Monthly Sales Trend')
    st.plotly_chart(fig_exec_trend, use_container_width=True)

    st.write("**Average Unit Price Trend**")
    price_trend = df.groupby(['year', 'month'])['unit_price'].mean().reset_index()
    price_trend['month_year'] = pd.to_datetime(price_trend['year'].astype(str) + '-' + price_trend['month'].astype(str))
    fig_price = px.line(price_trend, x='month_year', y='unit_price', title='Avg. Unit Price Over Time')
    st.plotly_chart(fig_price, use_container_width=True)

# --- Sales Team Tab ---
with tabs[1]:
    st.subheader("📈 Sales Trend for Selected Product")
    history_df = df[df["product_id"] == product_id].copy()
    history_df["month_year"] = pd.to_datetime(history_df["year"].astype(str) + "-" + history_df["month"].astype(str))
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

    delta = prediction - input_data['lag_1'].values[0]
    st.metric("📊 Change from Previous Month", f"{delta:+.2f}")

# --- Marketing Tab ---
with tabs[2]:
    st.subheader("🎯 Marketing Insights")
    holiday_sales = df.groupby("holiday")["total_quantity_sold"].mean()
    st.bar_chart(holiday_sales.rename({0: "Non-Holiday", 1: "Holiday"}))

    st.write("**Price Sensitivity**")
    price_scatter = px.scatter(df, x="unit_price", y="total_quantity_sold", trendline="ols",
                               title="Unit Price vs Quantity Sold")
    st.plotly_chart(price_scatter, use_container_width=True)

# --- Finance Tab ---
with tabs[3]:
    st.subheader("💰 Finance Overview")
    est_revenue = input_data['unit_price'].values[0] * prediction
    cost_ratio = input_data['freight_price'].values[0] / input_data['unit_price'].values[0]
    st.metric("Estimated Revenue", f"${est_revenue:,.2f}")
    st.metric("Freight Cost Ratio", f"{cost_ratio:.2f}")

# --- Supply Chain Tab ---
with tabs[4]:
    st.subheader("🚚 Inventory & Supply Chain")
    vol_df = df[df['product_id'] == product_id].copy()
    fig_vol = px.line(vol_df, x="month_index", y="total_quantity_sold", title="Demand Over Time")
    fig_vol.add_scatter(x=vol_df["month_index"], y=vol_df["lag_1"], mode="lines", name="Lag 1")
    st.plotly_chart(fig_vol, use_container_width=True)

# --- Model Explanation (SHAP) ---
with tabs[5]:
    st.subheader("🧠 Model Explanation (SHAP Values)")

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

    preprocessed_input = model.named_steps['preprocessor'].transform(input_data)
    feature_names = get_feature_names(model)
    explainer = shap.Explainer(model.named_steps['model'], feature_names=feature_names)
    shap_values = explainer(preprocessed_input)

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
