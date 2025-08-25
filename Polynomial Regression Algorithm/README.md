# 🔢 Polynomial Regression Algorithm
📌 Overview

This project demonstrates the implementation of Polynomial Regression (an extension of Linear Regression) using Python and Scikit-Learn. The dataset emp_sal.csv is used to predict employee salaries based on position levels.

Polynomial Regression is useful when the data shows a non-linear relationship between independent and dependent variables.

🚀 Features

Implementation of Simple Linear Regression for comparison

Implementation of Polynomial Regression (degree = 5)

Visualization of linear vs polynomial regression fit

Prediction of salary for a given position level (example: 6.5)

Model evaluation with R² Score

🛠️ Technologies Used

Python 🐍

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

📂 Dataset

File: emp_sal.csv

Columns:

Position Level

Salary

📜 Code Workflow

Import required libraries

Load dataset (emp_sal.csv)

Train Linear Regression model

Visualize results (red = actual data, blue = predicted line)

Apply PolynomialFeatures (degree = 5)

Train Polynomial Regression model

Visualize polynomial curve fit

Predict salary for a given level (example: 6.5)

📊 Visualizations

Linear Regression Fit (straight line)

Polynomial Regression Fit (curved line capturing non-linear trend)

📈 Example Prediction
lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)


📌 Output: 330378.78 (Predicted Salary for level 6.5)

🔮 Future Enhancements

Use GridSearchCV to select the best polynomial degree automatically

Apply Regularization (Ridge/Lasso) to avoid overfitting

Build an interactive Streamlit app for salary prediction
