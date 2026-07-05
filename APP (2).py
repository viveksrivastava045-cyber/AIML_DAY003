import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Page title
st.set_page_config(page_title="Life Insurance Predictor")

st.title("🏥 Life Insurance Prediction App")

st.write("Predict whether a person is likely to buy life insurance based on age.")

# Load dataset
df = pd.read_csv("insurance_data.csv")

# Features and target
X = df[["age"]]
y = df["bought_insurance"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# User input
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=25
)

# Prediction
if st.button("Predict"):

    prediction = model.predict([[age]])[0]
    probability = model.predict_proba([[age]])[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("✅ The person is likely to buy life insurance.")
    else:
        st.error("❌ The person is NOT likely to buy life insurance.")

    st.write(f"Probability of Buying Insurance: **{probability*100:.2f}%**")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)
