## 🧠 Project Title

**Sales Forecasting & Strategy Dashboard for Retail Products**

---------

## 📌 Overview

This project is an interactive Streamlit dashboard that forecasts product-level sales quantities using a machine learning pipeline. It helps business users simulate pricing decisions, understand sales trends, and optimize key levers across Sales, Marketing, Finance, and Supply Chain.

---

## 🎯 Objectives

* Predict monthly sales quantity based on business inputs (e.g., price, freight cost, holiday).
* Provide executives and stakeholders with a 360° view of key performance indicators.
* Offer explainability using SHAP to show how features affect predictions.
* Support decision-making across departments through visual analytics.

---

## 📂 Files Included

| File                        | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `Sales_Prediction_Dashboard.py` | Main Streamlit app (dashboard interface + logic)  |
| `retail_price.csv`          | Raw sales and pricing data used for visualization     |
| `processed_retail_data.csv` | Cleaned dataset with features for ML input            |
| `sales_pipeline.pkl`        | Trained scikit-learn pipeline for quantity prediction |

---

## ⚙️ Features

* **🎛 Sidebar Simulation**: Adjust unit price, freight cost, and holiday toggle to simulate scenarios
* **📈 Forecasting Model**: Predicts sales quantity using a trained ML pipeline
* **📊 Executive Overview**: KPIs, revenue growth, category performance
* **🚀 Sales Trends**: Monthly trends, customer volumes, and change from previous month
* **🎯 Marketing Analytics**: Price sensitivity, text quality impact, photo effect
* **💰 Finance View**: Revenue, freight cost ratios, profitability
* **🚚 Supply Chain View**: Volume trends, volatility, weight analysis
* **🧐 Explainability**: SHAP bar chart to interpret predictions

---

## 🧪 Tech Stack

* **Frontend**: Streamlit
* **Visualization**: Plotly, Matplotlib
* **Modeling**: Scikit-learn pipeline (`sales_pipeline.pkl`)
* **Explainability**: SHAP
* **Data**: Pandas, CSV files

---

## 🚀 How to Run

1. Clone the repo
2. Install requirements
3. Place `sales_pipeline.pkl`, `retail_price.csv`, and `processed_retail_data.csv` in the root folder
4. Run the app:

```bash
streamlit run Sales_Prediction_Dashboard.py
```

---

## 📊 Business Relevance & Impact

This project simulates **real-world retail decision-making** by providing an intuitive forecasting tool that visualizes how pricing strategies and seasonal factors influence sales performance. Designed for non-technical users, it enables cross-functional collaboration among **Sales, Marketing, Finance, and Supply Chain teams through a unified, data-driven platform**.

**By leveraging this tool, retail analysts, planners, and category managers can:**

- Make informed pricing decisions that maximize revenue

- Understand the key drivers behind sales trends and performance

- Align marketing initiatives with product attributes for better targeting

- Improve supply chain responsiveness by anticipating demand fluctuations

- Minimize waste and lost revenue through proactive inventory and sales planning

Ultimately, this empowers businesses to operate more efficiently and competitively in dynamic retail markets.
