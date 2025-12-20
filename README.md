## Churn Prediction System:
<b>A machine learning application for predicting customer churn using historical behavioral and transaction data.</b>
The system provides churn probability scores at the customer level, supports both single and bulk predictions, and includes interactive dashboards for data exploration, model evaluation, and business decision support. It is designed to help organizations identify at-risk customers and optimize retention strategies.

The models are trained on <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telecom Customer Churn</a> dataset.

## System Requirements:
1. Operating System: Windows / Linux / macOS
2. Python Version: 3.9+

## Dependencies:

# Standard Python Libraries
1. <code>logging</code>:- For logging processes.
2. <code>pathlib</code>:- For holding path of various required files.
3. <code>pickle</code>:- For storing python objects.

# Third-Party Libraries
1. <code>imblearn</code>:- For <code>SMOTE</code> oversampling.
2. <code>joblib</code>:- For storing trained models.
3. <code>pandas</code> and <code>numpy</code>:- For working with dataset and csv files.
4. <code>matplotlib</code> and <code>plotly</code>:- For dataset visualization and graphs.
5. <code>scikit-learn</code>:- For data preprocessing, classification models and benchmarking classification models.
6. <code>xgboost</code>:- For <code>XGBClassifier</code> classification model.
7. <code>streamlit</code>:- For accessing application frontend.

## Features
1. Supports configurable decision thresholds for churn prediction to balance precision and recall.
2. Enables experimentation with multiple classification models, including <code>XGBClassifier</code>, <code>Logistic Regression</code>, <code>SVC</code>, <code>DecisionTreeClassifier</code> and <code>RandomForestClassifier</code>.
3. Utilizes <code>RandomizedSearchCV</code> for hyperparameter optimization.
4. Provides interactive dashboards and in-depth exploratory analysis and model performance evaluation.
5. Maintains prediction and model evaluation history for analysis and comparison.
