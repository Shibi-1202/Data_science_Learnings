import streamlit as st
import numpy as np
import joblib

model = joblib.load("note_authenticator.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Banknote Authentication",
    layout="centered"
)

st.title("💵 Banknote Authentication System")
st.write("Use the model to classify whether a banknote is **Real** or **Fake** based on given features.")

st.markdown("---")

st.subheader("📥 Enter Banknote Features")

col1, col2 = st.columns(2)

with col1:
    variance = st.number_input("Variance of Wavelet Transformed Image", format="%.4f")
    curtosis = st.number_input("Curtosis of Wavelet Transformed Image", format="%.4f")

with col2:
    skewness = st.number_input("Skewness of Wavelet Transformed Image", format="%.4f")
    entropy = st.number_input("Entropy of Image", format="%.4f")


input_data = np.array([[variance, skewness, curtosis, entropy]])

if st.button("🔍 Predict"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    # Display Result
    st.markdown("### 🧾 Input Summary")
    st.write({
        "Variance": variance,
        "Skewness": skewness,
        "Curtosis": curtosis,
        "Entropy": entropy
    })

    st.markdown("---")

    if prediction == 0:
        st.success("🟢 **Result: The banknote is REAL.**")
    else:
        st.error("🔴 **Result: The banknote is FAKE.**")

st.markdown("---")
