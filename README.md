# 📘 Regression Assignment – Python + ML

This repository contains theory and practical examples for **Simple Linear Regression**, **Multiple Linear Regression**, and **Polynomial Regression** as part of the assignment from **Java + DSAPwskills**.

---

## 📌 Topics Covered

### 🔹 Simple Linear Regression
- ✅ What is Simple Linear Regression?
- ✅ Key Assumptions
- ✅ Equation: **Y = mX + c**
- ✅ Slope (m) and Intercept (c) calculation
- ✅ Least Squares Method
- ✅ R² Interpretation

### 🔹 Multiple Linear Regression
- ✅ Concept and Difference from Simple Linear Regression
- ✅ Key Assumptions
- ✅ Heteroscedasticity
- ✅ Multicollinearity handling
- ✅ Interaction Terms
- ✅ Categorical Variables Encoding

### 🔹 Regression Metrics
- ✅ Coefficient Significance
- ✅ Standard Error
- ✅ Adjusted R²
- ✅ R² Limitations

### 🔹 Polynomial Regression
- ✅ Use Case
- ✅ General Equation
- ✅ Differences from Linear Regression
- ✅ Model Selection Criteria (AIC, BIC, Cross-validation)
- ✅ Python Implementation

---

## 🧠 Sample Python Code

### ▶️ Polynomial Regression Example

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# Create a model
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X, y)

# Predictions
X_test = np.linspace(1, 5, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot
plt.scatter(X, y, color='red', label='Actual')
plt.plot(X_test, y_pred, color='blue', label='Prediction')
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
