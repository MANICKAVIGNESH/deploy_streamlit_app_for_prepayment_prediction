import streamlit as st
import numpy as np
from joblib import load

# Define or import CustomPipeline before loading the pickle file
class CustomPipeline:
    # your class implementation here
    pass

from joblib import load

try:
    elf = load('C:/Users/manic/Downloads/Combine_pipe_line.pkl')
    elf2 = load('C:/Users/manic/Downloads/Prepayment_risk_pipeline.pkl')
except Exception as e:
    st.error(f"An error occurred: {e}")

# Streamlit user inputs 
st.title("Mortgage Pre-payment and Pre-payment risk Prediction app") 

# Get all inputs from user at the beginning with unique keys
monthly_income = st.number_input("Enter the monthly income:", min_value=0, key='monthly_income')
OrigLoanTerm = st.number_input("Enter the loan term in months:", min_value=0, key='OrigLoanTerm')
OrigUPB = st.number_input("Enter the loan amount:", min_value=0, key='OrigUPB')
MonthsInRepayment = st.number_input("Enter the MonthsInRepayment value between 0 - 250:", min_value=0, max_value=250, key='MonthsInRepayment')
EMI = st.number_input("Enter the EMI value between 0 - 2500 dollars:", min_value=0, max_value=2500, key='EMI')
interest_amount = st.number_input("Enter the interest amount value between 50000 - 300000:", min_value=50000, max_value=300000, key='interest_amount')

if st.button("Predict"):  # Trigger prediction when the button is clicked
    # Prepare input data for classification model
    input_data = np.array([[monthly_income, OrigLoanTerm, OrigUPB]])

    st.write("Note: Convert 'Yes- delinquent' = 1 and 'No- delinquent' = 0")

    # Predict using the classification model
    try:
        classification_prediction = elf.clf.predict(input_data)
        st.write("Classification Prediction:", classification_prediction)

        # Predict with the regression model if classification result is 1 or 0
        if classification_prediction[0] in [0, 1]:
            regression_prediction = elf.reg.predict(input_data)
        else:
            regression_prediction = "It's delinquent"
        
        st.write("Regression Prediction:", regression_prediction)

        # Prepare second input data for the second classification model (pre-payment risk)
        input_data2 = np.array([[MonthsInRepayment, EMI, interest_amount]])

        # Predict with the second classification model
        classification_prediction2 = elf2.clf.predict(input_data2)

        st.write("Note: 1 = 'High' and 0 = 'Low'")
        st.write("Pre-Payment Risk:", classification_prediction2)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")