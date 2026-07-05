import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Life Insurance Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏥 Life Insurance Prediction App")

st.write("Predict whether a person is likely to buy life insurance based on age.")

# Load dataset
df = pd.read_csv("insurance_data.csv")
st.subheader("Dataset")
st.dataframe(df)


# Features and target
X = df[["age"]]
Y = df["bought_insurance"]

# Train model
model = LogisticRegression()
model.fit(X, Y)

# User input
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=20,
    step=1 
)

# Prediction
if st.button("Predict"):

    prediction = model.predict([[age]])[0]
    probability = model.predict_proba([[age]])[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("✅ YES.")
    else:
        st.error("❌ NO")

    st.write(f"Probability of Buying Insurance: **{probability*100:.2f}%**")


# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Model Details")

st.write("Coefficient:", model.coef_[0])
st.write("Intercept:", model.intercept_)
