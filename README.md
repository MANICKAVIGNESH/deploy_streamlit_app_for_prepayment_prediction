# deploy_streamlit_app_for_prepayment_prediction
# Mortgage Pre-payment and Risk Prediction App

This repository contains a Streamlit web application that predicts mortgage pre-payment amounts and assesses pre-payment risk using classification and regression models. The app processes user inputs and utilizes pre-trained machine learning models to deliver predictions.

## Features

- **EMI Calculation**: Calculates Equated Monthly Installment (EMI) based on loan terms, interest rates, and loan amounts.
- **Classification**: Predicts whether a mortgage is delinquent or not using logistic regression.
- **Regression**: For delinquent mortgages, predicts pre-payment amount based on monthly income and debt-to-income ratio (DTI).
- **Pre-payment Risk**: Assesses pre-payment risk using a second classification model.

## Technologies Used

- **Python**: The main programming language for this project.
- **Streamlit**: Used to build the interactive web application.
- **scikit-learn**: For classification and regression models.
- **joblib**: To load and save machine learning pipelines.
- **Google Drive API (gdown)**: For downloading the pre-trained models from Google Drive.

## Getting Started

### Prerequisites

- Python 3.x
- Streamlit
- gdown
- scikit-learn
- joblib
- numpy
- pandas

You can install the required dependencies by running:

```bash
pip install -r requirements.txt




