import streamlit as st

st.set_page_config(
    page_title="Superstore Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Superstore Sales Forecast Dashboard")

st.markdown("---")

st.markdown("""
Welcome to the **Superstore Sales Forecast Dashboard**.

This application was developed using **Streamlit** as part of the Sales Forecasting and Demand Analysis project.

### Features

- 📈 Sales Overview Dashboard
- 🔮 Sales Forecast Explorer
- ⚠️ Sales Anomaly Report
- 📦 Product Demand Segmentation

Use the navigation menu on the left to explore each section.
""")

st.markdown("---")

st.subheader("Project Summary")

col1, col2 = st.columns(2)

with col1:

    st.info("""
    **Machine Learning Models Used**

    - SARIMA
    - Facebook Prophet
    - XGBoost
    """)

with col2:

    st.success("""
    **Best Performing Model**

    ✅ XGBoost

    Selected based on:
    - Lowest MAE
    - Lowest RMSE
    - Lowest MAPE
    """)

st.markdown("---")

st.subheader("Model Performance")

st.table({

    "Model":[
        "SARIMA",
        "Prophet",
        "XGBoost"
    ],

    "MAE":[
        13524.25,
        10128.56,
        8621.00
    ],

    "RMSE":[
        17358.71,
        14561.39,
        11043.31
    ],

    "MAPE (%)":[
        22.56,
        14.33,
        12.41
    ]

})

st.markdown("---")

st.success("Use the sidebar to navigate through the dashboard pages.")
