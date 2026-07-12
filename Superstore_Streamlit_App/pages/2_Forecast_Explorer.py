import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Forecast Explorer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecast Explorer")

# -------------------------------------------------------
# Load Forecast Results
# -------------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_FILE = BASE_DIR / "forecast_results.csv"

forecast_df = pd.read_csv(FORECAST_FILE)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Forecast Options")

segment = st.sidebar.selectbox(
    "Select Category / Region",
    forecast_df["Segment"]
)

months = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=3,
    value=3
)

# -------------------------------------------------------
# Filter Selected Segment
# -------------------------------------------------------

selected = forecast_df[
    forecast_df["Segment"] == segment
]

# -------------------------------------------------------
# Forecast Table
# -------------------------------------------------------

st.subheader("Forecast Values")

if months == 1:
    table = selected[["Segment", "Month 1"]]

elif months == 2:
    table = selected[["Segment", "Month 1", "Month 2"]]

else:
    table = selected[
        ["Segment", "Month 1", "Month 2", "Month 3"]
    ]

st.dataframe(table, use_container_width=True)

# -------------------------------------------------------
# Forecast Chart
# -------------------------------------------------------

chart_df = selected.melt(
    id_vars="Segment",
    var_name="Month",
    value_name="Forecast Sales"
)

month_order = ["Month 1", "Month 2", "Month 3"]

chart_df["Month"] = pd.Categorical(
    chart_df["Month"],
    categories=month_order,
    ordered=True
)

chart_df = chart_df.iloc[:months]

fig = px.line(
    chart_df,
    x="Month",
    y="Forecast Sales",
    markers=True,
    title=f"{segment} Sales Forecast"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -------------------------------------------------------
# Model Performance
# -------------------------------------------------------

st.subheader("Best Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MAE",
        "8,620.99"
    )

with col2:
    st.metric(
        "RMSE",
        "11,043.31"
    )

st.info(
    "Forecast values are generated using the best-performing XGBoost model trained during the analysis phase."
)

# -------------------------------------------------------
# Download Forecast
# -------------------------------------------------------

csv = table.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast",
    data=csv,
    file_name="forecast_results.csv",
    mime="text/csv"
)
