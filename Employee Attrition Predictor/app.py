import streamlit as st
import pandas as pd
import pickle

st.title("Employee Attrition Predictor")

# Load model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input UI
satisfaction = st.slider("JobSatisfaction", 1, 4, 3)
income = st.number_input("MonthlyIncome", 1000, 20000, 5000)
age = st.slider("Age", 18, 60, 30)
years_at_company = st.slider("YearsAtCompany", 0, 40, 3)
distance = st.slider("DistanceFromHome", 1, 30, 10)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "JobSatisfaction": satisfaction,
        "MonthlyIncome": income,
        "Age": age,
        "YearsAtCompany": years_at_company,
        "DistanceFromHome": distance
    }])

    prediction = model.predict(input_data)[0]
    st.write("Prediction:", "❌ Likely to Leave" if prediction == 1 else "✅ Likely to Stay")
