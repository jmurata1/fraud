# fraud
# Credit Card Fraud Detection Project

## Overview
A machine learning project that predicts credit card fraud using a large transaction data. Our gradient boosting classifier achieves 92% recall and 86% precision in identifying fraudulent transactions, helping address the $33 billion global card fraud problem.

## Data
Our model includes the following features:
- category
- job_category
- hour_of_day
- distance_miles
- amt
To predict:
- is_fraud (target variable)

## Models
- Baseline: Logistic Regression
- Final Model: Gradient Boosting Classifier
  - ROC AUC Score: 0.99
  - Recall: 92%
  - Precision: 86%
  - Threshold: 0.3
 
## Reproducibility
- Refer to Google Colab Notebook for code with detailed instructions
  - https://colab.research.google.com/drive/1UM8_ngomZoiyERg1Pf-X90m13-yKgAn-#scrollTo=INBE34JuDCNn
- The code box labeled "Gradient Boosting Regressor using Optimal Threshold (0.3) for Better Recall" is the final model we decided on

## Business Recommendations
- Warning or require authentication for transactions above $650
- Time based verification and increased monitoring during hours with high fraud (10 PM - 3 AM)
- Pitch partnerships to large grocery stores and work together to implement card reader upgrades or reinforce employee training
- Online shopping should require multi-factor authentication for transactions
- Overall, we will provide this information to our customers and allow them to customize their risk profiles


