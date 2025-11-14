import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌲",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("RF_classify.pkl")

model = load_model()

st.title("🌲 Random Forest Classification App")
st.write("Provide the required input features to get the prediction.")


with st.form("prediction_form"):
    st.subheader("Input Features")

    p_class = st.selectbox("Passenger Class", [1, 2, 3])

    age = st.slider("Age", min_value=0, max_value=100, value=25)

    sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)

    parch = st.slider("Number of Parents/Children Aboard", 0, 10, 0)

    fare = st.slider("Ticket Fare ($)",0.0,600.0,32.0)

    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    submitted = st.form_submit_button("Predict 🚀")

def preprocess_input(pclass, age, sibsp, parch, fare, embarked):


    embarked_map = {"C": 0, "Q": 1, "S": 2}

    data = np.array([[
        pclass,
        age,
        sibsp,
        parch,
        fare,
        embarked_map[embarked]
    ]])

    return data

if submitted:
    input_data = preprocess_input(p_class, age, sibsp, parch, fare, embarked)
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("### 📝 Input Summary (Used for Prediction)")

    df_summary = pd.DataFrame({
        "Feature": ["Passenger Class", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
        "Value":   [p_class, age, sibsp, parch, fare, embarked]
    })

    st.table(df_summary)

    st.markdown("---")

    st.markdown("### 🎯 Prediction Result")
    if prediction == 1:
        st.success(f"**Survived** ✓ (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.error(f"**Not Survived** ✗ (Confidence: {prediction_proba[0]*100:.2f}%)")

    st.markdown("---")

    st.subheader("📊 Prediction Probabilities")
    st.write(f"- Survived: **{prediction_proba[1]:.4f}**")
    st.write(f"- Not Survived: **{prediction_proba[0]:.4f}**")

    

