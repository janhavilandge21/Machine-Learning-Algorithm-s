# 📈 Simple Linear Regression Algorithm

This project demonstrates the implementation of Linear Regression using the California Housing dataset from Scikit-Learn.

📌 Project Overview

Load the California Housing dataset (fetch_california_housing)

Explore dataset structure & features

Preprocess and split data into training/testing sets

Standardize features using StandardScaler

Train a Linear Regression model

Evaluate performance using:

Mean Squared Error (MSE)

R² Score (Coefficient of Determination)

Visualize prediction errors using Seaborn plots


Requirements (requirements.txt):

numpy
pandas
matplotlib
seaborn
scikit-learn

🚀 Usage

Example script:

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))

📊 Results

Achieved R² Score ≈ 0.34 (approx, based on current run)

Distribution of residuals plotted using Seaborn

MSE variance analyzed with cross-validation

📘 Dataset Information

Dataset: California Housing Dataset

Features:

MedInc → Median Income

HouseAge → Median House Age

AveRooms → Average Rooms per Household

AveBedrms → Average Bedrooms per Household

Population → Population per Block Group

AveOccup → Average Household Size

Latitude → Block Group Latitude

Longitude → Block Group Longitude

Target: MedHouseVal → Median House Value (in $100,000s)

📘 Learning Resources

Scikit-Learn: Linear Regression

Hands-On Machine Learning (O’Reilly)
