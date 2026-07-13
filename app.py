# ==========================================================
# End-to-End Sales Forecasting & Demand Intelligence System
# Streamlit Dashboard
# Author: Lathish Yadav
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Dashboard Title
# ----------------------------------------------------------

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

st.markdown(
"""
Interactive dashboard for

- Sales Analysis
- Forecasting
- Anomaly Detection
- Product Segmentation
"""
)

st.markdown("---")

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Week"] = df["Order Date"].dt.isocalendar().week
    df["Day"] = df["Order Date"].dt.day_name()

    df["Shipping Days"] = (
        df["Ship Date"] -
        df["Order Date"]
    ).dt.days

    return df


sales = load_data()

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Dashboard",

    [

        "Sales Overview",

        "Forecast Explorer",

        "Anomaly Report",

        "Demand Segments"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Dataset Loaded Successfully")

st.sidebar.write("Rows :", sales.shape[0])

st.sidebar.write("Columns :", sales.shape[1])

st.sidebar.write("Categories :", sales["Category"].nunique())

st.sidebar.write("Regions :", sales["Region"].nunique())

st.sidebar.markdown("---")

# ----------------------------------------------------------
# KPI Metrics
# ----------------------------------------------------------

total_sales = sales["Sales"].sum()

total_orders = sales["Order ID"].nunique()

avg_sales = sales["Sales"].mean()

shipping_days = sales["Shipping Days"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    f"${total_sales:,.2f}"
)

col2.metric(
    "Orders",
    total_orders
)

col3.metric(
    "Average Sales",
    f"${avg_sales:,.2f}"
)

col4.metric(
    "Average Shipping Days",
    round(shipping_days,2)
)

st.markdown("---")

# ==========================================================
# PAGE 1 : SALES OVERVIEW
# ==========================================================

if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")

    st.markdown("### Filter Data")

    col1, col2 = st.columns(2)

    with col1:
        selected_region = st.selectbox(
            "Select Region",
            ["All"] + sorted(sales["Region"].unique().tolist())
        )

    with col2:
        selected_category = st.selectbox(
            "Select Category",
            ["All"] + sorted(sales["Category"].unique().tolist())
        )

    # -----------------------------
    # Apply Filters
    # -----------------------------

    filtered = sales.copy()

    if selected_region != "All":
        filtered = filtered[
            filtered["Region"] == selected_region
        ]

    if selected_category != "All":
        filtered = filtered[
            filtered["Category"] == selected_category
        ]

    st.markdown("---")

    # -----------------------------
    # KPI Metrics
    # -----------------------------

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Total Sales",
        f"${filtered['Sales'].sum():,.2f}"
    )

    k2.metric(
        "Total Orders",
        filtered["Order ID"].nunique()
    )

    k3.metric(
        "Average Sales",
        f"${filtered['Sales'].mean():,.2f}"
    )

    st.markdown("---")

    # -----------------------------
    # Sales by Year
    # -----------------------------

    yearly = filtered.groupby("Year")["Sales"].sum().reset_index()

    fig = px.bar(
        yearly,
        x="Year",
        y="Sales",
        title="Total Sales by Year",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Monthly Sales TrendS
    # -----------------------------

    monthly = filtered.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"].sum().reset_index()

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Region
    # -----------------------------

    region_sales = filtered.groupby(
        "Region"
    )["Sales"].sum().reset_index()

    fig = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        title="Sales by Region"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Category
    # -----------------------------

    category_sales = filtered.groupby(
        "Category"
    )["Sales"].sum().reset_index()

    fig = px.pie(
        category_sales,
        names="Category",
        values="Sales",
        title="Category-wise Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Top 10 Products
    # -----------------------------

    top_products = filtered.groupby(
        "Product Name"
    )["Sales"].sum().sort_values(
        ascending=False
    ).head(10)

    st.subheader("🏆 Top 10 Products")

    fig = px.bar(
        top_products,
        orientation="h",
        title="Top 10 Products by Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Segment
    # -----------------------------

    segment_sales = filtered.groupby(
        "Segment"
    )["Sales"].sum().reset_index()

    fig = px.bar(
        segment_sales,
        x="Segment",
        y="Sales",
        color="Segment",
        title="Sales by Customer Segment"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Shipping Days
    # -----------------------------

    shipping = filtered.groupby(
        "Region"
    )["Shipping Days"].mean().reset_index()

    fig = px.bar(
        shipping,
        x="Region",
        y="Shipping Days",
        color="Region",
        title="Average Shipping Days"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Raw Dataset
    # -----------------------------

    st.subheader("Filtered Dataset")

    st.dataframe(filtered)

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_sales.csv",
        mime="text/csv"
    )
    
    # ==========================================================
# PAGE 2 : FORECAST EXPLORER
# ==========================================================

elif page == "Forecast Explorer":

    st.header("📈 Sales Forecast Explorer")

    st.markdown(
        "Forecast future sales using the Prophet forecasting model."
    )

    forecast_type = st.radio(
        "Forecast Type",
        ["Category", "Region"],
        horizontal=True
    )

    forecast_months = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # -----------------------------------------------------
    # Category Forecast
    # -----------------------------------------------------

    if forecast_type == "Category":

        selected = st.selectbox(
            "Select Category",
            sorted(sales["Category"].unique())
        )

        forecast_data = sales[
            sales["Category"] == selected
        ]

    # -----------------------------------------------------
    # Region Forecast
    # -----------------------------------------------------

    else:

        selected = st.selectbox(
            "Select Region",
            sorted(sales["Region"].unique())
        )

        forecast_data = sales[
            sales["Region"] == selected
        ]

    # -----------------------------------------------------
    # Monthly Sales
    # -----------------------------------------------------

    monthly = forecast_data.groupby(

        pd.Grouper(
            key="Order Date",
            freq="ME"
        )

    )["Sales"].sum().reset_index()

    monthly.columns = ["ds", "y"]

    st.subheader("Historical Monthly Sales")

    fig = px.line(

        monthly,

        x="ds",

        y="y",

        markers=True,

        title="Historical Sales"

    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # Prophet Model
    # -----------------------------------------------------

    model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

    model.fit(monthly)

    future = model.make_future_dataframe(

        periods=forecast_months,

        freq="ME"

    )

    forecast = model.predict(future)

    # -----------------------------------------------------
    # Forecast Chart
    # -----------------------------------------------------

    st.subheader("Forecast")

    prophet_fig = model.plot(forecast)

    st.pyplot(prophet_fig)

    # -----------------------------------------------------
    # Components
    # -----------------------------------------------------

    st.subheader("Trend & Seasonality")

    component_fig = model.plot_components(forecast)

    st.pyplot(component_fig)

    # -----------------------------------------------------
    # Forecast Table
    # -----------------------------------------------------

    future_table = forecast.tail(forecast_months)[

        [

            "ds",

            "yhat",

            "yhat_lower",

            "yhat_upper"

        ]

    ]

    future_table.columns = [

        "Forecast Date",

        "Predicted Sales",

        "Lower Bound",

        "Upper Bound"

    ]

    st.subheader("Forecast Values")

    st.dataframe(future_table)

    # -----------------------------------------------------
    # Download Forecast
    # -----------------------------------------------------

    csv = future_table.to_csv(index=False).encode("utf-8")

    st.download_button(

        "Download Forecast",

        csv,

        "forecast.csv",

        "text/csv"

    )

    # -----------------------------------------pip show seaborn------------
    # Model Metrics
    # -----------------------------------------------------

    st.subheader("Model Evaluation")

    metric1, metric2 = st.columns(2)

    metric1.metric(

        "MAE",

        "Replace with your MAE"

    )

    metric2.metric(

        "RMSE",

        "Replace with your RMSE"

    )

    st.info(
        """
        Note:

        Replace the MAE and RMSE values with the results
        obtained from your notebook (Task 3 Model Comparison).
        """
    )

    # -----------------------------------------------------
    # Forecast Summary
    # -----------------------------------------------------

    st.success(

        f"""
        Forecast generated successfully for **{selected}**.

        Forecast Horizon:
        **{forecast_months} Month(s)**

        Model Used:
        **Facebook Prophet**
        """

    )
    
    # ==========================================================
# PAGE 3 : ANOMALY REPORT
# ==========================================================

elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Detection")

    st.markdown(
        "Detect unusual sales spikes and drops using Isolation Forest and Z-Score."
    )

    # -----------------------------------------------------
    # Weekly Sales Aggregation
    # -----------------------------------------------------

    weekly = sales.groupby(

        pd.Grouper(
            key="Order Date",
            freq="W"
        )

    )["Sales"].sum().reset_index()

    # -----------------------------------------------------
    # Isolation Forest
    # -----------------------------------------------------

    iso = IsolationForest(

        contamination=0.05,

        random_state=42

    )

    weekly["Isolation"] = iso.fit_predict(

        weekly[["Sales"]]

    )

    anomaly_points = weekly[

        weekly["Isolation"] == -1

    ]

    # -----------------------------------------------------
    # Isolation Forest Plot
    # -----------------------------------------------------

    st.subheader("Isolation Forest Anomalies")

    fig = px.line(

        weekly,

        x="Order Date",

        y="Sales",

        title="Weekly Sales"

    )

    fig.add_scatter(

        x=anomaly_points["Order Date"],

        y=anomaly_points["Sales"],

        mode="markers",

        marker=dict(

            color="red",

            size=10

        ),

        name="Anomaly"

    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # Isolation Table
    # -----------------------------------------------------

    st.subheader("Detected Anomalies")

    st.dataframe(anomaly_points)

    # -----------------------------------------------------
    # Z Score Detection
    # -----------------------------------------------------

    weekly["Rolling Mean"] = weekly["Sales"].rolling(5).mean()

    weekly["Rolling Std"] = weekly["Sales"].rolling(5).std()

    weekly["Z Score"] = (

        weekly["Sales"] -

        weekly["Rolling Mean"]

    ) / weekly["Rolling Std"]

    zscore = weekly[

        weekly["Z Score"].abs() > 2

    ]

    # -----------------------------------------------------
    # Z Score Plot
    # -----------------------------------------------------

    st.subheader("Z-Score Anomalies")

    fig2 = px.line(

        weekly,

        x="Order Date",

        y="Sales",

        title="Weekly Sales with Z-Score"

    )

    fig2.add_scatter(

        x=zscore["Order Date"],

        y=zscore["Sales"],

        mode="markers",

        marker=dict(

            color="orange",

            size=10

        ),

        name="Z-Score"

    )

    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------
    # Z Score Table
    # -----------------------------------------------------

    st.subheader("Z-Score Outliers")

    st.dataframe(zscore)

    # -----------------------------------------------------
    # Compare Methods
    # -----------------------------------------------------

    st.subheader("Comparison")

    st.markdown("""

### Isolation Forest

- Machine Learning based

- Detects complex anomalies

- Works well with irregular sales patterns

---

### Z-Score

- Statistical method

- Detects values beyond ±2 Standard Deviations

- Easy to interpret

---

### Conclusion

Isolation Forest usually identifies more contextual anomalies,
whereas Z-Score highlights extreme deviations.

Using both methods together provides a more reliable anomaly detection strategy.

""")

    # -----------------------------------------------------
    # Download Reports
    # -----------------------------------------------------

    csv1 = anomaly_points.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Isolation Forest Report",

        csv1,

        "isolation_anomalies.csv",

        "text/csv"

    )

    csv2 = zscore.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Z-Score Report",

        csv2,

        "zscore_anomalies.csv",

        "text/csv"

    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(

        "Isolation Forest Anomalies",

        len(anomaly_points)

    )

    col2.metric(

        "Z-Score Anomalies",

        len(zscore)

    )
    
    # ==========================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTATION
# ==========================================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segmentation")

    st.markdown("""
    Segment products into different demand groups using
    **K-Means Clustering**.
    """)

    # ----------------------------------------
    # Aggregate Product Features
    # ----------------------------------------

    cluster_data = sales.groupby("Sub-Category").agg(

        Total_Sales=("Sales","sum"),

        Average_Sales=("Sales","mean"),

        Sales_Count=("Sales","count"),

        Shipping_Time=("Shipping Days","mean")

    ).reset_index()

    st.subheader("Aggregated Product Features")

    st.dataframe(cluster_data)

    # ----------------------------------------
    # Scaling
    # ----------------------------------------

    scaler = StandardScaler()

    scaled = scaler.fit_transform(

        cluster_data[

            [

                "Total_Sales",

                "Average_Sales",

                "Sales_Count",

                "Shipping_Time"

            ]

        ]

    )

    # ----------------------------------------
    # Elbow Method
    # ----------------------------------------

    inertia = []

    k_values = range(2,10)

    for k in k_values:

        model = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10

        )

        model.fit(scaled)

        inertia.append(model.inertia_)

    st.subheader("Elbow Method")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(

        list(k_values),

        inertia,

        marker="o"

    )

    ax.set_xlabel("Number of Clusters")

    ax.set_ylabel("Inertia")

    ax.set_title("Elbow Method")

    st.pyplot(fig)
    
        # ----------------------------------------
    # Apply K-Means Clustering
    # ----------------------------------------

    optimal_clusters = 4

    kmeans = KMeans(
        n_clusters=optimal_clusters,
        random_state=42,
        n_init=10
    )

    cluster_data["Cluster"] = kmeans.fit_predict(scaled)

    st.subheader("Clustered Product Data")

    st.dataframe(cluster_data)

    # ----------------------------------------
    # Assign Cluster Labels
    # ----------------------------------------

    cluster_names = {
        0: "High Volume",
        1: "Growing Demand",
        2: "Stable Demand",
        3: "Low Volume"
    }

    cluster_data["Demand Segment"] = cluster_data["Cluster"].map(cluster_names)

    st.subheader("Demand Segments")

    st.dataframe(

        cluster_data[
            [
                "Sub-Category",
                "Demand Segment",
                "Total_Sales",
                "Average_Sales"
            ]
        ]

    )

    # ----------------------------------------
    # PCA for Visualization
    # ----------------------------------------

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(scaled)

    pca_df = pd.DataFrame({

        "PCA1": pca_result[:,0],

        "PCA2": pca_result[:,1],

        "Cluster": cluster_data["Cluster"],

        "Demand Segment": cluster_data["Demand Segment"],

        "Sub-Category": cluster_data["Sub-Category"]

    })

    st.subheader("PCA Cluster Visualization")

    fig = px.scatter(

        pca_df,

        x="PCA1",

        y="PCA2",

        color="Demand Segment",

        hover_name="Sub-Category",

        title="Product Demand Segments"

    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------
    # Cluster Distribution
    # ----------------------------------------

    st.subheader("Cluster Distribution")

    distribution = cluster_data["Demand Segment"].value_counts()

    fig = px.bar(

        x=distribution.index,

        y=distribution.values,

        labels={

            "x":"Demand Segment",

            "y":"Number of Products"

        },

        title="Products in Each Demand Segment"

    )

    st.plotly_chart(fig, use_container_width=True)
    
        # ----------------------------------------
    # Cluster Summary
    # ----------------------------------------

    st.subheader("Cluster Summary")

    summary = cluster_data.groupby("Demand Segment").agg(

        Total_Sales=("Total_Sales","sum"),

        Average_Sales=("Average_Sales","mean"),

        Products=("Sub-Category","count")

    ).reset_index()

    st.dataframe(summary)

    # ----------------------------------------
    # Stocking Strategy
    # ----------------------------------------

    st.subheader("Recommended Stocking Strategy")

    strategy = pd.DataFrame({

        "Demand Segment":[

            "High Volume",

            "Growing Demand",

            "Stable Demand",

            "Low Volume"

        ],

        "Recommended Strategy":[

            "Maintain High Inventory Levels",

            "Increase Stock Gradually",

            "Maintain Current Inventory",

            "Reduce Inventory & Monitor Demand"

        ]

    })

    st.table(strategy)

    # ----------------------------------------
    # Cluster Statistics
    # ----------------------------------------

    st.subheader("Cluster Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Sub-Categories",
        cluster_data.shape[0]
    )

    c2.metric(
        "Clusters",
        optimal_clusters
    )

    c3.metric(
        "Largest Cluster",
        cluster_data["Demand Segment"].value_counts().idxmax()
    )

    # ----------------------------------------
    # Download Cluster Data
    # ----------------------------------------

    csv = cluster_data.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="⬇ Download Cluster Report",

        data=csv,

        file_name="product_clusters.csv",

        mime="text/csv"

    )

    # ----------------------------------------
    # Business Insights
    # ----------------------------------------

    st.subheader("Business Insights")

    st.success("""

✔ High Volume products should always be available in inventory.

✔ Growing Demand products require proactive replenishment.

✔ Stable Demand products should follow normal stocking policies.

✔ Low Volume products should be stocked carefully to reduce inventory costs.

✔ Product segmentation helps optimize warehouse utilization and improves forecasting accuracy.

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
### 📈 End-to-End Sales Forecasting & Demand Intelligence System

**Internship Project**

Developed by **Lathish Yadav**

Technologies Used:

- Python
- Streamlit
- Pandas
- Plotly
- Prophet
- Scikit-Learn
- K-Means Clustering
- Isolation Forest
- PCA
- Machine Learning
"""
)

# ==========================================================
# PAGE FOOTER & PROJECT INFORMATION
# ==========================================================

st.markdown("---")

with st.expander("📘 Project Information"):

    st.write("""
### End-to-End Sales Forecasting & Demand Intelligence System

This dashboard was developed as part of the Data Science Internship Project.

### Features

✔ Sales Dashboard

✔ Time Series Forecasting

✔ Prophet Forecast Model

✔ Isolation Forest Anomaly Detection

✔ Z-Score Outlier Detection

✔ K-Means Product Segmentation

✔ PCA Visualization

✔ Interactive Plotly Charts

✔ Download Reports

### Technologies

• Python

• Pandas

• NumPy

• Plotly

• Streamlit

• Prophet

• Scikit-Learn

• PCA

• K-Means

• Isolation Forest

""")

st.markdown("---")

st.caption(
    "Developed by Lathish Yadav | Data Science Internship Project | 2026"
)
