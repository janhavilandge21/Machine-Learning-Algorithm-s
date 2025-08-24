# 🔗 Ridge & Lasso Regression

This project demonstrates the implementation of Ridge Regression and Lasso Regression on the California Housing dataset using Scikit-Learn. Both techniques are used to handle multicollinearity and overfitting in linear regression by applying regularization.

📌 Project Overview

Load and explore the California Housing dataset

Implement Linear Regression baseline model

Apply Ridge Regression (L2 regularization)

Apply Lasso Regression (L1 regularization)

Tune hyperparameters (alpha) using GridSearchCV

Compare performance using Mean Squared Error (MSE) and R² Score

Visualize residuals with Seaborn plots


Requirements (requirements.txt):

numpy
pandas
matplotlib
seaborn
scikit-learn

🚀 Usage

Example code:

import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# Evaluation
print("Ridge R² Score:", r2_score(y_test, ridge_pred))
print("Lasso R² Score:", r2_score(y_test, lasso_pred))

📊 Results

GridSearchCV was used to find the optimal alpha values

Ridge performed best at alpha ≈ 55 with MSE ≈ -0.557

Lasso also tuned with similar range, showing slightly higher bias

Residual distribution visualized using histplot/distplot

📘 Dataset Information

Dataset: California Housing Dataset

Features: Median Income, House Age, Average Rooms, Bedrooms, Population, Household Size, Latitude, Longitude

Target: Median House Value (in $100,000s)

📘 Learning Resources

Scikit-Learn: Ridge Regression

Scikit-Learn: Lasso Regression

Hands-On ML with Scikit-Learn, Keras, and TensorFlow
