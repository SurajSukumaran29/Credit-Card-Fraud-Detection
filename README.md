# Credit Card Fraud Detection

This repository contains a Credit Card Fraud Detection project where machine learning models are used to predict whether a transaction is fraudulent or legitimate. The project employs Random Forest and XGBoost models, achieving 86% accuracy with Random Forest and 88% accuracy with XGBoost.
Table of Contents

    Project Overview
    Installation
    Data
    Model Training
    Model Evaluation
    Visualizations
    License

Project Overview

In this project, we use machine learning techniques to detect fraudulent credit card transactions based on various features such as transaction amount, user behavior, and geographical information. The goal is to create a model that accurately classifies transactions as fraudulent or legitimate.

Key Features:

    Preprocessed and cleaned transaction data
    Feature engineering for better prediction accuracy
    Trained Random Forest and XGBoost models
    Performance evaluation using accuracy, ROC curve, confusion matrix, and precision-recall curve

Installation

To run this project locally, follow these steps:

    Clone the repository:

git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

Install the required dependencies:

pip install -r requirements.txt

If you don't have the requirements.txt file, you can manually install the necessary packages:

    pip install pandas numpy scikit-learn xgboost matplotlib seaborn

Data

The dataset used for training the models consists of transaction data with features such as:

    TransactionAmount: The amount of the transaction
    CardLimit: The cardholder's credit limit
    UserAge: The age of the cardholder
    UserIncome: The income of the cardholder
    TransactionHour: The hour of the transaction
    IsOnlineTransaction: Whether the transaction was online
    IsWeekendTransaction: Whether the transaction occurred on the weekend
    ...and more.

The data was preprocessed to handle missing values, categorical features, and scaled numeric features before being fed into the machine learning models.

Note: The dataset is not included in this repository for privacy and licensing reasons. You can use publicly available datasets like the Kaggle Credit Card Fraud Detection dataset or any similar dataset.
Model Training

The project includes training two models for fraud detection:

    Random Forest Classifier
    XGBoost Classifier

The models were trained using the cleaned and preprocessed transaction data, and key performance metrics like accuracy, confusion matrix, ROC curve, and precision-recall curve were evaluated.

# Example code for model training
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Instantiate the models
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5)
xgb_model = XGBClassifier()

# Train the models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

Model Evaluation

The performance of both models was evaluated using the following metrics:

    Accuracy: Percentage of correct predictions.
    Confusion Matrix: True positives, false positives, true negatives, and false negatives.
    ROC Curve: Visualizes the model’s performance across different thresholds.
    Precision-Recall Curve: Assesses the trade-off between precision and recall, important in imbalanced datasets like fraud detection.

Here is an example of how you can evaluate the model's performance:

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

Visualizations

The following visualizations were created to evaluate the model:

    Feature Importance: Bar plot showing the importance of each feature for the model's prediction.
    Confusion Matrix: Heatmap visualizing the performance of the model.
    ROC Curve: Curve showing the trade-off between true positive rate and false positive rate.
    Precision-Recall Curve: Shows the precision and recall for different thresholds.
