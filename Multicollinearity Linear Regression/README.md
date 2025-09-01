# Multicollinearity in Linear Regression
📌 Overview

This project demonstrates the concept of Multicollinearity in Linear Regression using Python. It explores how correlated independent variables (predictors) can affect the regression model, interpretation of coefficients, and model accuracy.

Two datasets are used:

Advertising Dataset – Predicting sales using TV, Radio, and Newspaper advertising budgets.

Salary Dataset – Predicting Salary using Years of Experience and Age.

📂 Dataset

Advertising.csv

Features: TV, Radio, Newspaper

Target: Sales

Salary_Data.csv

Features: YearsExperience, Age

Target: Salary

🧑‍💻 Code Workflow

Import required libraries (pandas, numpy, statsmodels, matplotlib, seaborn).

Load dataset into Pandas DataFrame.

Perform Exploratory Data Analysis (EDA).

Check correlation matrix to detect multicollinearity.

Build OLS (Ordinary Least Squares) regression model using statsmodels.

Interpret results (coefficients, p-values, R², F-statistic, VIF).

Compare models with and without multicollinearity.

📊 Key Results
Advertising Dataset

TV and Radio are significant predictors.

Newspaper is not significant (p-value = 0.860).

R² ≈ 0.897 → Strong model fit.

Salary Dataset

YearsExperience and Age are highly correlated (correlation = 0.987).

Multicollinearity inflates the standard errors → misleading p-values.

Despite high R² ≈ 0.960, interpretation of coefficients becomes unreliable.

🚩 Conclusion

Multicollinearity does not affect model accuracy (R²), but it distorts the interpretation of coefficients.

Always check correlation matrix and Variance Inflation Factor (VIF) before trusting regression results.

Feature selection or dimensionality reduction can help mitigate multicollinearity.

📌 Technologies Used

Python (pandas, numpy, statsmodels, matplotlib, seaborn)

Jupyter Notebook
