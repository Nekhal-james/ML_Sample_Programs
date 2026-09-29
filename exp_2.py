import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ---- Step 1: Load and Clean Dataset ----
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

# Correctly parse whitespace, handle missing values denoted by '?', and drop them
df = pd.read_csv(url, sep=r'\s+', names=columns)
df = df.replace('?', np.nan).dropna()

# ---- Step 2: Select Features & Split ----
X = df[["displacement"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 3: Train Linear Regression ----
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# ---- Step 4: Train Polynomial Regression (Degree 2) ----
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# ---- Step 5: Evaluation ----
mse_lr, r2_lr = mean_squared_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)
mse_poly, r2_poly = mean_squared_error(y_test, y_pred_poly), r2_score(y_test, y_pred_poly)

print("--- Linear Regression Performance ---")
print(f"MSE : {mse_lr:.2f}")
print(f"R2  : {r2_lr:.3f}")

print("\n--- Polynomial Regression Performance ---")
print(f"MSE : {mse_poly:.2f}")
print(f"R2  : {r2_poly:.3f}")

# ---- Step 6: Split Visualization into Side-by-Side Plots ----
# Generate smooth lines tracking from min to max displacement
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Create a figure with 1 row and 2 columns sharing the same Y-axis scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# 1. Left Plot: Linear Regression
ax1.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
ax1.plot(X_line, lr_model.predict(X_line), color='red', linewidth=2, label='Linear Fit')
ax1.set_title(f'Linear Fit\nMSE: {mse_lr:.2f} | R²: {r2_lr:.3f}')
ax1.set_xlabel('Displacement')
ax1.set_ylabel('MPG')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# 2. Right Plot: Polynomial Regression
ax2.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
ax2.plot(X_line, poly_model.predict(poly.transform(X_line)), color='green', linewidth=2, label='Poly Fit')
ax2.set_title(f'Polynomial Fit (Degree 2)\nMSE: {mse_poly:.2f} | R²: {r2_poly:.3f}')
ax2.set_xlabel('Displacement')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()