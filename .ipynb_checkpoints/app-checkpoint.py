import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="K-Means Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation using K-Means")
st.write("Interactive app for **2D & 3D clustering**, real-time prediction, and batch analysis.")

st.divider()

# ----------------------------------
# Sidebar – Model Selection
# ----------------------------------
st.sidebar.header("⚙️ Model Configuration")

model_type = st.sidebar.radio(
    "Select Clustering Model",
    ("2D K-Means (Income, Spending)", "3D K-Means (Age, Income, Spending)")
)

# ----------------------------------
# Load Models
# ----------------------------------
if model_type == "2D K-Means (Income, Spending)":
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    dimensions = "2D"
else:
    kmeans = joblib.load("kmeans_3d_model.pkl")
    scaler = joblib.load("scaler_3d.pkl")
    dimensions = "3D"

# ----------------------------------
# Customer Personas
# ----------------------------------
cluster_personas = {
    0: "💼 **High Income – Low Spending**\n\nCautious customers with strong purchasing power.",
    1: "🛍️ **High Income – High Spending**\n\nPremium customers generating high revenue.",
    2: "🎯 **Average Income – Average Spending**\n\nBalanced and stable customers.",
    3: "💸 **Low Income – High Spending**\n\nImpulse buyers with high engagement.",
    4: "📉 **Low Income – Low Spending**\n\nPrice-sensitive, low engagement customers."
}

# ----------------------------------
# Section 1 – Individual Prediction
# ----------------------------------
st.header("🔍 Individual Customer Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 30)

with col2:
    income = st.slider("Annual Income (k$)", 10, 150, 60)

with col3:
    spending = st.slider("Spending Score (1–100)", 1, 100, 50)

if st.button("Predict Segment"):
    
    if dimensions == "2D":
        input_data = np.array([[income, spending]])
    else:
        input_data = np.array([[age, income, spending]])

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Customer Segment: **Cluster {cluster}**")
    st.info(cluster_personas.get(cluster, "Cluster description unavailable."))

st.divider()

# ----------------------------------
# Section 2 – Cluster Visualization
# ----------------------------------
st.header("📊 Cluster Visualization")

if dimensions == "2D":
    st.subheader("2D Clustering View")

    # Dummy grid for visualization
    income_range = np.linspace(10, 150, 100)
    spending_range = np.linspace(1, 100, 100)

    grid = np.array([[i, s] for i in income_range for s in spending_range])
    grid_scaled = scaler.transform(grid)
    labels = kmeans.predict(grid_scaled)

    df_plot = pd.DataFrame(grid, columns=["Income", "Spending"])
    df_plot["Cluster"] = labels

    fig = px.scatter(
        df_plot,
        x="Income",
        y="Spending",
        color="Cluster",
        title="2D K-Means Clustering"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("3D Clustering View")

    age_range = np.linspace(18, 70, 20)
    income_range = np.linspace(10, 150, 20)
    spending_range = np.linspace(1, 100, 20)

    grid = np.array([[a, i, s] for a in age_range for i in income_range for s in spending_range])
    grid_scaled = scaler.transform(grid)
    labels = kmeans.predict(grid_scaled)

    df_plot = pd.DataFrame(grid, columns=["Age", "Income", "Spending"])
    df_plot["Cluster"] = labels

    fig = px.scatter_3d(
        df_plot,
        x="Age",
        y="Income",
        z="Spending",
        color="Cluster",
        title="3D K-Means Clustering"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------
# Section 3 – Batch Prediction (CSV)
# ----------------------------------
st.header("📂 Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

st.write("Required columns:")
if dimensions == "2D":
    st.code("Annual Income (k$), Spending Score (1-100)")
else:
    st.code("Age, Annual Income (k$), Spending Score (1-100)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        if dimensions == "2D":
            X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
        else:
            X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

        X_scaled = scaler.transform(X)
        df["Predicted Cluster"] = kmeans.predict(X_scaled)
        df["Customer Persona"] = df["Predicted Cluster"].map(cluster_personas)

        st.success("Batch prediction completed!")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results",
            csv,
            "cluster_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
