import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model, model_columns = pickle.load(open('model.pkl', 'rb'))

st.title("Enterprise Customer Churn Predictor")
st.write("Predict whether a customer is likely to cancel their subscription based on account metrics.")

# Sidebar inputs for user
st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 150.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 10.0, 8000.0, 1000.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

# Process inputs
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'PaymentMethod': [payment_method]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Prediction button
if st.button("Predict Churn Risk"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: Customer is likely to churn. (Probability: {prediction_proba[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk: Customer is likely to stay. (Probability: {prediction_proba[0][0]*100:.1f}%)")
