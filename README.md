# 📊 Superstore Sales Forecasting Dashboard

## Overview

This project is an interactive Streamlit dashboard developed for sales analysis, forecasting, anomaly detection, and product demand segmentation using the Superstore Sales dataset. It combines data analytics, machine learning, and visualization techniques to help users understand historical sales trends and make data-driven decisions.

## Features

* 📈 Interactive Sales Overview Dashboard
* 🔮 Sales Forecasting using XGBoost
* 🚨 Anomaly Detection using Isolation Forest
* 📦 Product Demand Segmentation using K-Means Clustering
* 📊 Interactive visualizations with Plotly
* 📥 Download forecast results as CSV

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Plotly
* Joblib
* Matplotlib

## Machine Learning Models

### Sales Forecasting

* XGBoost Regressor

### Anomaly Detection

* Isolation Forest

### Product Segmentation

* K-Means Clustering

## Project Structure

```text
Superstore_Streamlit_App/
│
├── app.py
├── requirements.txt
├── xgboost_model.pkl
│
├── data/
│   └── Superstore.csv
│
├── pages/
│   ├── 1_Sales_Overview.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Anomaly_Report.py
│   └── 4_Product_Demand_Segments.py
│
└── assets/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Superstore-Sales-Forecasting-Dashboard.git
```

Move into the project directory:

```bash
cd Superstore-Sales-Forecasting-Dashboard
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Dashboard Pages

### 1. Sales Overview

* Total sales summary
* Monthly sales trend
* Sales by region
* Sales by category
* Interactive filters

### 2. Forecast Explorer

* Sales forecasting for 1–3 months
* Category-wise forecast
* Region-wise forecast
* Model performance (MAE and RMSE)
* Download forecast results

### 3. Anomaly Report

* Detect unusual sales patterns
* Visualize anomalies
* View anomaly dates and sales values

### 4. Product Demand Segments

* Customer demand clustering
* Cluster visualization
* Sub-category cluster mapping
* Demand insights

## Dataset

The project uses the **Superstore Sales Dataset**, which contains information such as:

* Order Date
* Sales
* Category
* Sub-Category
* Region
* Customer Details
* Product Information

## Future Enhancements

* Deploy the dashboard on Streamlit Community Cloud
* Add Prophet and SARIMA forecasting models
* Include real-time sales data integration
* Build advanced business KPI dashboards
* Add user authentication

## Author

**Tanishka Gandhi**

Third Year Computer Engineering Student

D. Y. Patil College of Engineering, Akurdi


