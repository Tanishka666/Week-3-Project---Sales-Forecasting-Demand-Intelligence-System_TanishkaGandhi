import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecast Explorer", layout="wide")

st.title("📈 Forecast Explorer")

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = pd.read_csv("data/Sales.csv", encoding="latin1")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)

monthly_sales = (
    df.groupby("Order Date")["Sales"]
    .sum()
    .resample("M")
    .sum()
    .reset_index()
)

monthly_sales.columns = ["Date", "Sales"]

# -------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------
model = joblib.load("xgboost_model.pkl")

# -------------------------------------------------------
# Create Features
# -------------------------------------------------------
data = monthly_sales.copy()

data["Month"] = data["Date"].dt.month
data["Quarter"] = data["Date"].dt.quarter

# Create Season Feature
def get_season(month):
    if month in [10,11,12, 1]:
        return 1
    elif month in [2,3, 4, 5]:
        return 2
    elif month in [6, 7, 8,9]:
        return 3
    else:
        return 4

data["Season"] = data["Month"].apply(get_season)

data["Lag_1"] = data["Sales"].shift(1)
data["Lag_2"] = data["Sales"].shift(2)
data["Lag_3"] = data["Sales"].shift(3)

data["Rolling_Mean"] = (
    data["Sales"]
    .rolling(window=3)
    .mean()
)

data.dropna(inplace=True)

feature_cols = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_Mean",
    "Month",
    "Quarter",
    "Season"
]

X = data[feature_cols]
y = data["Sales"]

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Forecast Options")

forecast_type = st.sidebar.selectbox(
    "Forecast By",
    [
        "Overall",
        "Category",
        "Region"
    ]
)

months = st.sidebar.slider(
    "Forecast Horizon (Months)",
    1,
    3,
    3
)

filtered_df = df.copy()

# -------------------------------------------------------
# Category Filter
# -------------------------------------------------------
if forecast_type == "Category":

    category = st.sidebar.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered_df = df[df["Category"] == category]

# -------------------------------------------------------
# Region Filter
# -------------------------------------------------------
elif forecast_type == "Region":

    region = st.sidebar.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    filtered_df = df[df["Region"] == region]

# -------------------------------------------------------
# Monthly Aggregation
# -------------------------------------------------------
series = (
    filtered_df.groupby("Order Date")["Sales"]
    .sum()
    .resample("M")
    .sum()
    .reset_index()
)

series.columns = ["Date", "Sales"]

# -------------------------------------------------------
# Prepare Forecast Data
# -------------------------------------------------------

forecast_data = series.copy()

forecast_data["Month"] = forecast_data["Date"].dt.month
forecast_data["Quarter"] = forecast_data["Date"].dt.quarter
forecast_data["Season"] = forecast_data["Month"].apply(get_season)
forecast_data["Lag_1"] = forecast_data["Sales"].shift(1)
forecast_data["Lag_2"] = forecast_data["Sales"].shift(2)
forecast_data["Lag_3"] = forecast_data["Sales"].shift(3)

forecast_data["Rolling_Mean"] = (
    forecast_data["Sales"]
    .rolling(3)
    .mean()
)

forecast_data.dropna(inplace=True)

history = forecast_data.copy()

future_predictions = []

last_date = history.iloc[-1]["Date"]

# -------------------------------------------------------
# Recursive Forecast
# -------------------------------------------------------

for i in range(months):

    lag1 = history.iloc[-1]["Sales"]
    lag2 = history.iloc[-2]["Sales"]
    lag3 = history.iloc[-3]["Sales"]

    rolling_mean = np.mean([lag1, lag2, lag3])

    future_date = last_date + pd.DateOffset(months=1)

    month = future_date.month
    quarter = future_date.quarter

    season = get_season(month)

    future_features = pd.DataFrame(
    [[
        lag1,
        lag2,
        lag3,
        rolling_mean,
        month,
        quarter,
        season
    ]],
    columns=feature_cols
)

    prediction = model.predict(future_features)[0]

    future_predictions.append(
        {
            "Date": future_date,
            "Forecast Sales": prediction
        }
    )

    history.loc[len(history)] = [
    future_date,
    prediction,
    month,
    quarter,
    season,
    lag1,
    lag2,
    lag3,
    rolling_mean
]

    last_date = future_date

forecast_df = pd.DataFrame(future_predictions)

# -------------------------------------------------------
# Forecast Table
# -------------------------------------------------------

st.subheader("Forecast Values")

display_df = forecast_df.copy()
display_df["Forecast Sales"] = (
    display_df["Forecast Sales"]
    .round(2)
)

st.dataframe(
    display_df,
    use_container_width=True
)

# -------------------------------------------------------
# Forecast Chart
# -------------------------------------------------------

st.subheader("Forecast Visualization")

fig = go.Figure()

# Historical Sales
fig.add_trace(
    go.Scatter(
        x=series["Date"],
        y=series["Sales"],
        mode="lines+markers",
        name="Historical Sales"
    )
)

# Forecast Sales
fig.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast Sales"],
        mode="lines+markers",
        name="Forecast",
        line=dict(dash="dash")
    )
)

fig.update_layout(
    title="Sales Forecast",
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# Model Performance
# -------------------------------------------------------

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MAE",
        f"{mae:,.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

# -------------------------------------------------------
# Download Forecast
# -------------------------------------------------------

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast CSV",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

# -------------------------------------------------------
# Forecast Summary
# -------------------------------------------------------

st.subheader("Forecast Summary")

for _, row in forecast_df.iterrows():

    st.write(
        f"**{row['Date'].strftime('%B %Y')}** : "
        f"₹ {row['Forecast Sales']:,.2f}"
    )

st.info(
    "Forecasts are generated using the trained XGBoost model "
    "with lag features and rolling mean."
)
