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

with tabs[0]:
    st.subheader("📊 Executive Overview")

    # Aggregate data for current and last year
    latest_year = df_full['year'].max()
    last_year = latest_year - 1

    # Total units sold and total revenue for latest year
    this_year_qty = df_full[df_full['year'] == latest_year]['qty'].sum()
    this_year_revenue = df_full[df_full['year'] == latest_year]['total_price'].sum()
    this_year_avg_price = df_full[df_full['year'] == latest_year]['unit_price'].mean()

    # Same for last year
    last_year_qty = df_full[df_full['year'] == last_year]['qty'].sum()
    last_year_revenue = df_full[df_full['year'] == last_year]['total_price'].sum()

    # YoY growth in volume (qty)
    growth_qty = ((this_year_qty - last_year_qty) / last_year_qty) * 100 if last_year_qty != 0 else 0
    # YoY growth in revenue
    growth_revenue = ((this_year_revenue - last_year_revenue) / last_year_revenue) * 100 if last_year_revenue != 0 else 0

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Units Sold (YTD)", f"{this_year_qty:,}")
    col2.metric("Total Revenue (YTD)", f"${this_year_revenue:,.2f}")
    col3.metric("Avg Unit Price (YTD)", f"${this_year_avg_price:.2f}")
    col4.metric("YoY Volume Growth", f"{growth_qty:.2f}%")

    st.markdown("---")

    # Top 5 Categories by Revenue
    cat_revenue = df_full[df_full['year'] == latest_year].groupby("product_category_name")["total_price"].sum().sort_values(ascending=False).head(5)
    st.write("### 🏆 Top 5 Product Categories by Revenue")
    st.bar_chart(cat_revenue)
    st.caption("Categories generating the highest revenue this year")

    st.markdown("---")

    # Top 5 Products by product_score (if available)
    if 'product_score' in df_full.columns:
        top_products = (
            df_full[df_full['year'] == latest_year]
            .groupby('product_id')['product_score']
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        st.write("### 🌟 Top 5 Products by Product Score")
        st.bar_chart(top_products)
        st.caption("Products with highest average scores")

    # Overall monthly sales trend (qty) - keep your previous chart for trend
    trend_df = df_full.groupby(['year', 'month'])['qty'].sum().reset_index()
    trend_df['month_year'] = pd.to_datetime(trend_df['year'].astype(str) + '-' + trend_df['month'].astype(str))
    fig_exec_trend = px.line(trend_df, x='month_year', y='qty', title='Overall Monthly Sales Trend')
    st.plotly_chart(fig_exec_trend, use_container_width=True)
    st.caption("Shows monthly sales trend across all products")

with tabs[1]:
    st.subheader("📈 Sales Execution Insights")

    # Filter data for the selected product
    product_sales = df_full[df_full["product_id"] == product_id].copy()
    product_sales["month_year"] = pd.to_datetime(product_sales["year"].astype(str) + "-" + product_sales["month"].astype(str))
    product_sales = product_sales.sort_values("month_year")

    # Monthly Sales Trend (qty)
    fig_sales_trend = px.line(
        product_sales,
        x="month_year",
        y="qty",
        markers=True,
        title=f"Monthly Sales Trend for Product ID: {product_id}",
        labels={"qty": "Quantity Sold", "month_year": "Month"}
    )
    st.plotly_chart(fig_sales_trend, use_container_width=True)
    st.caption("Monthly sales quantity for the selected product")

    # Customer Volume Trend
    if 'customers' in product_sales.columns:
        cust_trend = product_sales.groupby("month_year")["customers"].sum().reset_index()
        fig_cust_trend = px.line(
            cust_trend,
            x="month_year",
            y="customers",
            title="Customer Volume Trend",
            labels={"customers": "Number of Customers", "month_year": "Month"}
        )
        st.plotly_chart(fig_cust_trend, use_container_width=True)
        st.caption("Monthly customer count trend for the selected product")

    # Sales by Weekday vs Weekend
    if {'year', 'month', 'day'}.issubset(df_full.columns):
        product_sales["date"] = pd.to_datetime(product_sales[["year", "month", "day"]])
        product_sales["weekday"] = product_sales["date"].dt.weekday
        product_sales["is_weekend"] = product_sales["weekday"].apply(lambda x: 1 if x >= 5 else 0)
        sales_weekday = product_sales.groupby("is_weekend")["qty"].sum().rename({0: "Weekday", 1: "Weekend"})
        st.write("### 🕘 Sales by Weekday vs Weekend")
        st.bar_chart(sales_weekday)
        st.caption("Sales distribution between weekdays and weekends")

    # Volume vs Unit Price Correlation Scatter Plot
    fig_vol_price = px.scatter(
        product_sales,
        x="unit_price",
        y="qty",
        trendline="ols",
        title="Volume vs Unit Price Correlation",
        labels={"unit_price": "Unit Price", "qty": "Quantity Sold"}
    )
    st.plotly_chart(fig_vol_price, use_container_width=True)

    # Change from Previous Month Metric (Prediction vs Lag_1)
    delta = prediction - input_data['lag_1'].values[0]
    st.metric("📋 Change from Previous Month", f"{delta:+.2f}")

with tabs[2]:
    st.subheader("🎯 Marketing Insights")

    # Holiday vs Non-Holiday Sales
    # Create a unified Holiday vs Non-Holiday grouping
    df_full['holiday_group'] = df_full['holiday'].apply(lambda x: 'Holiday' if x > 0 else 'Non-Holiday')

    # Group by the new holiday_group column and get mean qty
    holiday_sales = df_full.groupby('holiday_group')['qty'].mean()
# Plot it
    st.write("### 🧨 Holiday vs Non-Holiday Sales")
    st.bar_chart(holiday_sales)
    st.caption("Average sales quantity during holidays vs non-holidays")

    # Price Sensitivity Plot (unit_price vs qty)
    fig_price_sensitivity = px.scatter(
        df_full,
        x="unit_price",
        y="qty",
        trendline="ols",
        title="Unit Price vs Quantity Sold (Price Sensitivity)",
        labels={"unit_price": "Unit Price", "qty": "Quantity Sold"}
    )
    st.plotly_chart(fig_price_sensitivity, use_container_width=True)

    # Marketing Text Quality: Product Description Length (your original snippet)
    st.write("**Marketing Text Quality: Product Description Length**")
    desc_len = df_full.groupby("product_description_lenght")["qty"].mean().reset_index()
    fig_desc = px.scatter(
        desc_len,
        x="product_description_lenght",
        y="qty",
        title="Product Description Length vs Avg. Quantity Sold",
        labels={"product_description_lenght": "Description Length (chars)", "qty": "Avg. Quantity Sold"}
    )
    st.plotly_chart(fig_desc, use_container_width=True)
    st.caption("Longer descriptions can improve customer interest")

    # Text Features Impact (optional, if you want to keep product_name_length)
    if 'product_name_length' in df_full.columns:
        name_len = df_full.groupby("product_name_length")["qty"].mean().reset_index()
        fig_name_len = px.scatter(
            name_len,
            x="product_name_length",
            y="qty",
            title="Product Name Length vs Avg. Quantity Sold",
            labels={"product_name_length": "Name Length (chars)", "qty": "Avg. Quantity Sold"}
        )
        st.plotly_chart(fig_name_len, use_container_width=True)
        st.caption("Longer product names can improve customer interest")

    # Photo Count vs Sales
    if 'product_photos_qty' in df_full.columns:
        photo_qty = df_full.groupby("product_photos_qty")["qty"].mean().reset_index()
        fig_photos = px.bar(
            photo_qty,
            x="product_photos_qty",
            y="qty",
            title="Avg. Sales by Number of Product Photos",
            labels={"product_photos_qty": "Number of Photos", "qty": "Avg. Quantity Sold"}
        )
        st.plotly_chart(fig_photos, use_container_width=True)
        st.caption("More photos can influence customer purchases")


# --- Finance Tab ---
with tabs[3]:
    st.subheader("💰 Finance Overview")

    est_revenue = input_data['unit_price'].values[0] * prediction
    cost_ratio = input_data['freight_price'].values[0] / input_data['unit_price'].values[0]
    st.metric("Estimated Revenue", f"${est_revenue:,.2f}")
    st.metric("Freight Cost Ratio", f"{cost_ratio:.2f}")

  
    # Revenue Breakdown by Category
    st.subheader("💼 Revenue by Product Category")
    revenue_by_category = df_full.groupby('product_category_name')['total_price'].sum().sort_values(ascending=False)
    st.bar_chart(revenue_by_category)

    # Freight Price vs Revenue Scatter Plot
    st.subheader("💸 Freight Price vs Total Revenue")
    fig_freight_vs_revenue = px.scatter(
        df_full,
        x='freight_price',
        y='total_price',
        title='Freight Price vs Total Revenue',
        labels={'freight_price': 'Freight Price', 'total_price': 'Total Revenue'}
    )
    st.plotly_chart(fig_freight_vs_revenue, use_container_width=True)

    # Freight-to-Revenue Ratio Over Time
    st.subheader("🚛 Freight-to-Revenue Ratio Over Time")
    df_full['freight_ratio'] = df_full['freight_price'] / df_full['total_price']
    freight_ratio_time = df_full.groupby('month_year')['freight_ratio'].mean()
    st.line_chart(freight_ratio_time)

    # Average Unit Price Over Time
    st.subheader("📈 Average Unit Price Over Time")
    avg_price_month = df_full.groupby('month_year')['unit_price'].mean()
    st.line_chart(avg_price_month)


# --- Supply Chain Tab ---
with tabs[4]:
    st.header("🚚 Supply Chain Overview")

    # Volume Trend Over Time
    st.subheader("📦 Volume Trend Over Time")
    volume_trend = df_full.groupby('month_year')['qty'].sum()
    st.line_chart(volume_trend)

    # Weight vs Volume Demand Scatter Plot
    st.subheader("⚖️ Product Weight vs Sales Volume")
    fig_weight_vs_qty = px.scatter(
        df_full,
        x='product_weight_g',
        y='qty',
        title='Product Weight vs Quantity Sold',
        labels={'product_weight_g': 'Product Weight (g)', 'qty': 'Quantity Sold'}
    )
    st.plotly_chart(fig_weight_vs_qty, use_container_width=True)

    # Volume Forecast Comparison (Predicted vs Historical) - placeholder
    st.subheader("🔮 Volume Forecast Comparison")
    st.info("Add your forecast vs historical comparison chart here")

    # Category-Level Demand Volatility
    st.subheader("📉 Demand Volatility by Product Category")
    volatility = df_full.groupby('product_category_name')['qty'].std().sort_values(ascending=False)
    st.bar_chart(volatility)


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
