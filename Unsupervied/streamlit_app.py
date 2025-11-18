import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="K-Means Clustering - Inference", layout="wide")

st.title("K-Means Cluster Prediction")
st.write("Enter feature values to get the cluster number.")

cluster_labels = {
    0: "Budget Customers",
    1: "Standard Shoppers",
    2: "Target Customers (High Income & Spending)",
    3: "Potential Customers (High Income, Low Spending)",
    4: "Low Income, High Spending"
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("K_means_cluster.pkl")
    scaler = joblib.load("Scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()


st.subheader("Input Features")

f1 = st.slider("Income in (k$) 1",0,150,25)
f2 = st.slider("Spending Score (1 - 100)",0,100,30)



# Collect all inputs
input_data = np.array([[f1, f2]])

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Cluster"):
    scaled = scaler.transform(input_data)
    cluster = model.predict(scaled)[0]

    st.success(f"🎯 **Predicted Cluster: {cluster} - {cluster_labels.get(cluster, 'Unknown')} **")
