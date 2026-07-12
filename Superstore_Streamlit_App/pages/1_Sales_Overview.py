# ==========================================================
# Import Libraries
# ==========================================================
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Sales Overview",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Overview Dashboard")

# ==========================================================
# Load Dataset
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "sales.csv"

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    return df

df = load_data()

# ==========================================================
# Create Time Features
# ==========================================================

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# ==========================================================
# Sidebar Filters
# ==========================================================

st.sidebar.header("Filters")

selected_region = st.sidebar.multiselect(
    "Select Region",
    options=sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

selected_category = st.sidebar.multiselect(
    "Select Category",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

# ==========================================================
# Apply Filters
# ==========================================================

filtered_df = df[
    (df["Region"].isin(selected_region)) &
    (df["Category"].isin(selected_category))
]

# ==========================================================
# KPI Cards
# ==========================================================

total_sales = filtered_df["Sales"].sum()

total_orders = filtered_df["Order ID"].nunique()

average_sales = filtered_df["Sales"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${total_sales:,.2f}")

col2.metric("Total Orders", total_orders)

col3.metric("Average Order Value", f"${average_sales:,.2f}")

st.markdown("---")

# ==========================================================
# Total Sales by Year
# ==========================================================

st.subheader("Total Sales by Year")

yearly_sales = (
    filtered_df
    .groupby("Year")["Sales"]
    .sum()
    .reset_index()
)

fig_year = px.bar(
    yearly_sales,
    x="Year",
    y="Sales",
    text_auto=".2s",
    title="Yearly Sales"
)

st.plotly_chart(fig_year, use_container_width=True)

# ==========================================================
# Monthly Sales Trend
# ==========================================================

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig_month = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig_month, use_container_width=True)

# ==========================================================
# Sales by Region
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Sales by Region")

    region_sales = (
        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig_region = px.pie(
        region_sales,
        names="Region",
        values="Sales"
    )

    st.plotly_chart(fig_region, use_container_width=True)

# ==========================================================
# Sales by Category
# ==========================================================

with col2:

    st.subheader("Sales by Category")

    category_sales = (
        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        text_auto=".2s"
    )

    st.plotly_chart(fig_category, use_container_width=True)

# ==========================================================
# Show Dataset
# ==========================================================

st.markdown("---")

st.subheader("Filtered Sales Data")

st.dataframe(filtered_df)
