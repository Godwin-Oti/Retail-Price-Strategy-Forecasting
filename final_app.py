import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

# --- Load Data & Model ---
df_full = pd.read_csv("retail_price.csv")       # Full dataset for EDA/visuals
df_model = pd.read_csv("processed_retail_data.csv")            # Dataset with features for model input
model = joblib.load("sales_pipeline.pkl")

# Compute month_index in df_full
df_full["month_index"] = (df_full["year"] - df_full["year"].min()) * 12 + df_full["month"]


# --- Title ---
st.title("📦 Sales Quantity Prediction Dashboard")

# --- Sidebar: Product & Month Selection ---
st.sidebar.header("🔍 Select Product & Month")

product_id = st.sidebar.selectbox("Product ID", df_full['product_id'].unique())

months = list(range(1, 13))
month_names = [pd.to_datetime(str(m), format='%m').strftime('%B') for m in months]
selected_month_name = st.sidebar.selectbox("Month", month_names)
month = months[month_names.index(selected_month_name)]

# --- Sidebar: Simulation Controls ---
st.sidebar.header("⚙️ Adjust Simulation Inputs")

if 'unit_price' not in df_model.columns or 'freight_price' not in df_model.columns:
    st.error("Missing required columns 'unit_price' or 'freight_price' in model input data.")
    st.stop()

unit_price = st.sidebar.slider(
    "Unit Price",
    float(df_model['unit_price'].min()),
    float(df_model['unit_price'].max()),
    float(df_model['unit_price'].mean())
)
freight_price = st.sidebar.slider(
    "Freight Price",
    float(df_model['freight_price'].min()),
    float(df_model['freight_price'].max()),
    float(df_model['freight_price'].mean())
)
holiday_flag = st.sidebar.selectbox("Holiday", [0, 1])

# --- Prepare Input for Prediction ---
product_df = df_model[df_model['product_id'] == product_id]

if product_df.empty:
    st.error("No historical data for selected product to infer features.")
    st.stop()

max_year = product_df['year'].max()
max_month = product_df[product_df['year'] == max_year]['month'].max()
max_month_index = product_df['month_index'].max()

months_diff = (month - max_month) if month >= max_month else (12 - max_month + month)
month_index = max_month_index + months_diff

lag_1_row = product_df[(product_df['year'] == max_year) & (product_df['month'] == max_month)]
lag_1 = lag_1_row['lag_1'].values[0] if not lag_1_row.empty else 0

prod_cat = product_df['product_category_name'].mode()
if prod_cat.empty:
    st.error("Product category not found for selected product.")
    st.stop()
product_category_name = prod_cat.values[0]

input_data = pd.DataFrame([{
    'product_category_name': product_category_name,
    'year': max_year,
    'month': month,
    'month_index': month_index,
    'lag_1': lag_1,
    'unit_price': unit_price,
    'freight_price': freight_price,
    'holiday': holiday_flag
}])

# --- Prediction ---
prediction = model.predict(input_data)[0]
st.metric("📈 Predicted Quantity Sold", f"{prediction:.2f}")

# --- Tabs ---
tabs = st.tabs(["Executive", "Sales", "Marketing", "Finance", "Supply Chain", "Model Explanation"])

# --- Executive Summary ---
with tabs[0]:
    st.subheader("📊 Executive Overview")

    this_year = df_full[df_full['year'] == df_full['year'].max()]['qty'].sum()
    last_year = df_full[df_full['year'] == df_full['year'].max() - 1]['qty'].sum()
    growth = ((this_year - last_year) / last_year) * 100 if last_year != 0 else 0

    st.metric("Total Units Sold (YTD)", f"{this_year:,}")
    st.metric("Year-over-Year Growth", f"{growth:.2f}%")

    cat_summary = df_full.groupby("product_category_name")["qty"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(cat_summary)
    st.caption("Top 5 product categories by total units sold")

    trend_df = df_full.groupby(['year', 'month'])['qty'].sum().reset_index()
    trend_df['month_year'] = pd.to_datetime(trend_df['year'].astype(str) + '-' + trend_df['month'].astype(str))
    fig_exec_trend = px.line(trend_df, x='month_year', y='qty', title='Overall Monthly Sales Trend')
    st.plotly_chart(fig_exec_trend, use_container_width=True)
    st.caption("Shows monthly sales trend across all products")

    st.write("**Average Unit Price Trend**")
    price_trend = df_full.groupby(['year', 'month'])['unit_price'].mean().reset_index()
    price_trend['month_year'] = pd.to_datetime(price_trend['year'].astype(str) + '-' + price_trend['month'].astype(str))
    fig_price = px.line(price_trend, x='month_year', y='unit_price', title='Avg. Unit Price Over Time')
    st.plotly_chart(fig_price, use_container_width=True)
    st.caption("Tracks average product price over time")

# --- Sales Team Tab ---
with tabs[1]:
    st.subheader("📈 Sales Trend for Selected Product")

    history_df = df_full[df_full["product_id"] == product_id].copy()
    history_df["month_year"] = pd.to_datetime(history_df["year"].astype(str) + "-" + history_df["month"].astype(str))
    history_df = history_df.sort_values("month_year")

    fig_trend = px.line(
        history_df,
        x="month_year",
        y="qty",
        markers=True,
        title=f"Sales Trend for Product ID: {product_id}",
        labels={"qty": "Quantity Sold", "month_year": "Month"}
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Monthly sales quantity for the selected product")

    delta = prediction - input_data['lag_1'].values[0]
    st.metric("📋 Change from Previous Month", f"{delta:+.2f}")

    st.write("### 🏖️ Holiday vs Non-Holiday Sales")
    holiday_sales = df_full.groupby("holiday")["qty"].mean().rename({0: "Non-Holiday", 1: "Holiday"})
    st.bar_chart(holiday_sales)
    st.caption("Average sales quantity during holidays vs non-holidays")

# --- Marketing Tab ---
with tabs[2]:
    st.subheader("🎯 Marketing Insights")

    st.write("**Visual Appeal: Product Photos Quantity**")
    photo_qty = df_full.groupby("product_photos_qty")["qty"].mean().reset_index()
    fig_photos = px.bar(photo_qty, x="product_photos_qty", y="qty", 
                       title="Avg. Sales by Number of Product Photos",
                       labels={"product_photos_qty": "Number of Photos", "qty": "Avg. Quantity Sold"})
    st.plotly_chart(fig_photos, use_container_width=True)
    st.caption("Higher number of photos may influence sales")

    st.write("**Marketing Text Quality: Product Description Length**")
    desc_len = df_full.groupby("product_description_lenght")["qty"].mean().reset_index()
    fig_desc = px.scatter(desc_len, x="product_description_lenght", y="qty", 
                          title="Product Description Length vs Avg. Quantity Sold",
                          labels={"product_description_lenght": "Description Length (chars)", "qty": "Avg. Quantity Sold"})
    st.plotly_chart(fig_desc, use_container_width=True)
    st.caption("Longer descriptions can improve customer interest")

    st.write("**Price Sensitivity**")
    price_scatter = px.scatter(df_full, x="unit_price", y="qty", trendline="ols",
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

    vol_df = df_full[df_full['product_id'] == product_id].copy()
    fig_vol = px.line(vol_df, x="month_index", y="qty", title="Demand Over Time")
    fig_vol.add_scatter(x=vol_df["month_index"], y=vol_df["lag_price"], mode="lines", name="Lag 1")
    st.plotly_chart(fig_vol, use_container_width=True)

# --- Model Explanation (SHAP) ---
with tabs[5]:
    st.subheader("🧐 Model Explanation (SHAP Values)")

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

    st.subheader("🔎 Model Input")
    st.write(input_data.drop(columns=['year']).reset_index(drop=True))
