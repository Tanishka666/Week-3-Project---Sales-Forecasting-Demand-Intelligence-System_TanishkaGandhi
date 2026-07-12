import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Product Demand Segments",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Demand Segmentation")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/sales.csv",
        encoding="latin1"
    )

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    return df


df = load_data()

# ---------------------------------------------------
# Create Year
# ---------------------------------------------------

df["Year"] = df["Order Date"].dt.year

# ---------------------------------------------------
# Total Sales
# ---------------------------------------------------

total_sales = (
    df.groupby("Sub-Category")["Sales"]
      .sum()
)

# ---------------------------------------------------
# Average Order Value
# ---------------------------------------------------

average_order = (
    df.groupby("Sub-Category")["Sales"]
      .mean()
)

# ---------------------------------------------------
# Monthly Sales
# ---------------------------------------------------

monthly_sales = (
    df.groupby(
        [
            "Sub-Category",
            pd.Grouper(
                key="Order Date",
                freq="M"
            )
        ]
    )["Sales"]
    .sum()
)

# ---------------------------------------------------
# Volatility
# ---------------------------------------------------

volatility = monthly_sales.groupby(
    "Sub-Category"
).std()

# ---------------------------------------------------
# Growth Rate
# ---------------------------------------------------

yearly_sales = (
    df.groupby(
        [
            "Sub-Category",
            "Year"
        ]
    )["Sales"]
    .sum()
    .unstack()
)

growth_rate = (
    (
        yearly_sales.iloc[:, -1]
        -
        yearly_sales.iloc[:, 0]
    )
    /
    yearly_sales.iloc[:, 0]
) * 100

# ---------------------------------------------------
# Feature Table
# ---------------------------------------------------

cluster_data = pd.DataFrame({

    "Total Sales": total_sales,

    "Growth Rate": growth_rate,

    "Volatility": volatility,

    "Average Order Value": average_order

})

cluster_data = cluster_data.fillna(0)

# ---------------------------------------------------
# Scale Data
# ---------------------------------------------------

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_data)

# ---------------------------------------------------
# KMeans
# ---------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_data["Cluster"] = kmeans.fit_predict(
    scaled_data
)

# ---------------------------------------------------
# Labels
# ---------------------------------------------------

cluster_labels = {

    0: "Growing Demand",

    1: "Low Volume, Stable Demand",

    2: "High Volume, Stable Demand"

}

cluster_data["Demand Group"] = (
    cluster_data["Cluster"]
    .map(cluster_labels)
)

# ---------------------------------------------------
# PCA
# ---------------------------------------------------

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(
    scaled_data
)

pca_df = pd.DataFrame({

    "PC1": pca_data[:, 0],

    "PC2": pca_data[:, 1],

    "Demand Group": cluster_data["Demand Group"]

})

# ---------------------------------------------------
# Scatter Plot
# ---------------------------------------------------

st.subheader("Demand Cluster Visualization")

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "Growing Demand": "blue",
    "Low Volume, Stable Demand": "green",
    "High Volume, Stable Demand": "red"
}

for group in pca_df["Demand Group"].unique():

    temp = pca_df[
        pca_df["Demand Group"] == group
    ]

    ax.scatter(

        temp["PC1"],

        temp["PC2"],

        label=group,

        s=120,

        color=colors[group]

    )

ax.set_title(
    "K-Means Product Demand Segments"
)

ax.set_xlabel("Principal Component 1")

ax.set_ylabel("Principal Component 2")

ax.grid(True)

ax.legend()

st.pyplot(fig)

# ---------------------------------------------------
# Table
# ---------------------------------------------------

st.subheader(
    "Sub-Category Demand Groups"
)

display = cluster_data.copy()

display.reset_index(inplace=True)

display.rename(

    columns={

        "index": "Sub-Category"

    },

    inplace=True

)

st.dataframe(

    display,

    use_container_width=True

)

# ---------------------------------------------------
# Cluster Summary
# ---------------------------------------------------

st.subheader(
    "Demand Group Summary"
)

summary = (
    display.groupby(
        "Demand Group"
    )[
        [
            "Total Sales",
            "Growth Rate",
            "Volatility",
            "Average Order Value"
        ]
    ]
    .mean()
)

st.dataframe(
    summary.round(2),
    use_container_width=True
)

# ---------------------------------------------------
# Stocking Strategy
# ---------------------------------------------------

st.subheader(
    "Recommended Stocking Strategy"
)

st.markdown("""

### 📈 Growing Demand

- Increase inventory gradually.
- Monitor sales trends regularly.
- Plan future stock expansion.

---

### 📦 High Volume, Stable Demand

- Maintain high inventory.
- Replenish stock frequently.
- Prioritize these products.

---

### 📉 Low Volume, Stable Demand

- Keep limited inventory.
- Avoid overstocking.
- Reorder only when required.

""")
