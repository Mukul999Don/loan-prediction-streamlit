import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("loan_data.csv")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

model = RandomForestClassifier()
model.fit(X, y)

st.title("🏦 Intelligent Loan Approval System")

# Inputs
age = st.slider("Age", 20, 60)
income = st.slider("Income", 20000, 120000)
credit = st.slider("Credit Score", 300, 850)
loan_amount = st.slider("Loan Amount", 50000, 500000)
loan_term = st.selectbox("Loan Term (months)", [12,18,24,36])
existing = st.selectbox("Existing Loan", [0,1])
exp = st.slider("Employment Years", 0, 30)

# Prediction
input_data = pd.DataFrame([[age,income,credit,loan_amount,loan_term,existing,exp]],
                         columns=X.columns)

if st.button("Predict Loan Approval"):
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
