import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Balance Scale Prediction", layout="centered")

# Load model
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("⚖️ Balance Scale Prediction App")
st.write("Enter the values below to predict whether the scale balances Left, Right, or Balanced.")

# Input sliders
LW = st.slider("Left Weight", 1, 5, 3)
LD = st.slider("Left Distance", 1, 5, 3)
RW = st.slider("Right Weight", 1, 5, 3)
RD = st.slider("Right Distance", 1, 5, 3)

# Format input
input_data = np.array([[LW, LD, RW, RD]])
scaled_input = scaler.transform(input_data)

# Prediction button
if st.button("Predict"):
    pred = model.predict(scaled_input)[0]

    st.subheader("🔍 Prediction Result:")
    st.success(f"The Scale is **{pred}ight sided**")

    # Show what inputs model used
    st.subheader("📌 Inputs Used")
    st.json({
        "Left Weight": LW,
        "Left Distance": LD,
        "Right Weight": RW,
        "Right Distance": RD
    })
