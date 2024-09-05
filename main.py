import streamlit as st
import numpy as np
from joblib import load
import gdown

# Define or import CustomPipeline before loading the pickle file
class CustomPipeline:
    # your class implementation here
    pass
    
# Google Drive download URL
download_url = 'https://drive.google.com/uc?id=1zCTevk022HBD9lLUns4fxR7u0M1Ii9HM'

# Destination path for the downloaded file
output = 'combined_pipeline.pkl'

# Download the file from Google Drive
gdown.download(download_url, output, quiet=False)

# Load the pipeline
try:
    elf = load(output)
except Exception as e:
    st.error(f"An error occurred: {e}")

# Streamlit user inputs 
st.title("Mortgage Prediction App") 

# Input fields for the user
monthly_income = st.number_input("Monthly Income", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
loan_amount = st.number_input("Loan Amount", min_value=0)

input_data = np.array([[monthly_income, loan_term, loan_amount]])

if st.button("Predict"):
    # Predict with the classification model
    try:
        classification_prediction = elf.clf.predict(input_data)
        st.write("Classification Prediction(Note - '0' mean NO-prepayment and '1' mean Prepayment):", classification_prediction)

        if classification_prediction[0] == 1:
            # Follow the original code for classification result 1
            regression_prediction = elf.reg.predict(input_data)
            st.write("Regression Prediction:", regression_prediction)
        elif classification_prediction[0] == 0:
            # Follow the alternate code for classification result 0
            regression_prediction_alt = elf.reg.predict(input_data)
            st.write("Regression Prediction (Alt):", regression_prediction_alt)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
