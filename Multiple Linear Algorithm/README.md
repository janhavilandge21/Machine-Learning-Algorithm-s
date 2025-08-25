# 📊 Multiple Linear Regression Algorithm
📌 Overview

This project demonstrates the implementation of Multiple Linear Regression using Python and Scikit-Learn. The dataset used is 50_Startups.csv, where the goal is to predict company profits based on R&D Spend, Administration, Marketing Spend, and State.

The model is trained, evaluated, and tested for performance with visualization support.

🚀 Features

Data preprocessing (handling categorical variables with One-Hot Encoding)

Splitting dataset into training and testing sets

Implementing Multiple Linear Regression with scikit-learn

Model evaluation using R² Score & Mean Squared Error

Visualization of predictions vs actual values

Saving and reusing trained models with Pickle

🛠️ Technologies Used

Python 🐍

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

📂 Dataset

File: 50_Startups.csv

Columns:

R&D Spend

Administration

Marketing Spend

State (Categorical → converted to dummy variables)

Profit (Target Variable)

📈 Model Performance

R² Score: 0.9347 (93.47%)

📜 Code Workflow

Import required libraries

Load dataset (50_Startups.csv)

Preprocess data (dummy variables for categorical features)

Split into train/test sets

Train Multiple Linear Regression model

Make predictions on test data

Evaluate model using R² Score

Save trained model with Pickle

📊 Visualizations

Scatter plots of predictions vs actual values

Distribution plots of residuals

🔮 Future Enhancements

Implement Ridge & Lasso Regression for regularization

Apply Cross Validation for better evaluation

Deploy model using Streamlit / Flask


