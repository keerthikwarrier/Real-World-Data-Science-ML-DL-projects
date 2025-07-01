
# Your Streamlit code here...
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_model.pkl")

st.title("Employee Attrition Predictor")

satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)

# Add more inputs...

if st.button("Predict"):
    input_df = pd.DataFrame([[satisfaction, monthly_income]], columns=["JobSatisfaction", "MonthlyIncome"])
    prediction = model.predict(input_df)
    st.write("Prediction:", "Likely to Leave" if prediction[0] == 1 else "Likely to Stay")
