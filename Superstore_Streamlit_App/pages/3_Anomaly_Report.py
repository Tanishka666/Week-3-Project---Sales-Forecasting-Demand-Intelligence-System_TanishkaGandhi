import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Anomaly Report",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Sales Anomaly Report")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/sales.csv", encoding="latin1")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    return df


df = load_data()

# ---------------------------------------------------
# Weekly Sales
# ---------------------------------------------------

weekly_sales = (
    df.groupby("Order Date")["Sales"]
      .sum()
      .resample("W")
      .sum()
      .reset_index()
)

# ---------------------------------------------------
# Isolation Forest
# ---------------------------------------------------

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly_sales["Anomaly"] = model.fit_predict(
    weekly_sales[["Sales"]]
)

anomalies = weekly_sales[
    weekly_sales["Anomaly"] == -1
]

# ---------------------------------------------------
# Chart
# ---------------------------------------------------

st.subheader("Weekly Sales with Detected Anomalies")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(
    weekly_sales["Order Date"],
    weekly_sales["Sales"],
    label="Weekly Sales"
)

ax.scatter(
    anomalies["Order Date"],
    anomalies["Sales"],
    color="red",
    s=100,
    label="Anomaly"
)

ax.set_title("Isolation Forest Anomaly Detection")

ax.set_xlabel("Order Date")

ax.set_ylabel("Sales")

ax.legend()

ax.grid(True)

st.pyplot(fig)

# ---------------------------------------------------
# Table
# ---------------------------------------------------

st.subheader("Detected Anomalies")

display = anomalies[[
    "Order Date",
    "Sales"
]].copy()

display["Sales"] = display["Sales"].round(2)

st.dataframe(
    display,
    use_container_width=True
)

# ---------------------------------------------------
# Total Anomalies
# ---------------------------------------------------

st.metric(
    "Total Anomalies",
    len(display)
)

# ---------------------------------------------------
# Business Explanation
# ---------------------------------------------------

st.subheader("Business Interpretation")

st.markdown("""

### Possible Reasons for High Sales

- Festival season
- Promotional campaigns
- Bulk customer orders
- Seasonal demand

### Possible Reasons for Low Sales

- Low customer demand
- Inventory shortage
- Public holidays
- Operational issues

""")

# ---------------------------------------------------
# Download
# ---------------------------------------------------

csv = display.to_csv(index=False).encode("utf-8")

st.download_button(

    "📥 Download Anomaly Report",

    csv,

    "Anomaly_Report.csv",

    "text/csv"

)
