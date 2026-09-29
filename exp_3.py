import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target 

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standard Linear Regression
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 4. Ridge Regression with Hyperparameter Tuning
alphas = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(Ridge(), alphas, cv=5).fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)

# 5. Lasso Regression with Hyperparameter Tuning
lasso_cv = GridSearchCV(Lasso(), alphas, cv=5).fit(X_train, y_train)
y_pred_lasso = lasso_cv.predict(X_test)

# Extract scores for evaluation
mse_lr, r2_lr = mean_squared_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)
mse_ridge, r2_ridge = mean_squared_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)
mse_lasso, r2_lasso = mean_squared_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)

metrics = {
    "Linear": [mse_lr, r2_lr, "N/A"],
    "Ridge": [mse_ridge, r2_ridge, ridge_cv.best_params_['alpha']],
    "Lasso": [mse_lasso, r2_lasso, lasso_cv.best_params_['alpha']]
}

# 6. Print Results Table
print("\n" + "="*55)
print(f"{'Model':<12} | {'MSE':<12} | {'R2 Score':<12} | {'Best Alpha':<10}")
print("="*55)
for model_name, values in metrics.items():
    print(f"{model_name:<12} | {values[0]:<12.2f} | {values[1]:<12.4f} | {str(values[2]):<10}")
print("="*55)

# ---- Step 7: Visual Comparison using Bar Graphs with Value Labels ----
models = list(metrics.keys())
mse_values = [metrics[m][0] for m in models]
r2_values = [metrics[m][1] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Mean Squared Error (Lower is better)
colors1 = ['salmon', 'tomato', 'darkred']
bars1 = ax1.bar(models, mse_values, color=colors1, edgecolor='black', width=0.5)
ax1.set_title('Mean Squared Error (Lower is Better)')
ax1.set_ylabel('MSE Value')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
# Add values on top of MSE bars (rounded to 1 decimal place)
ax1.bar_label(bars1, fmt='%.1f', padding=3)

# Plot 2: R2 Score (Higher is better)
colors2 = ['skyblue', 'dodgerblue', 'navy']
bars2 = ax2.bar(models, r2_values, color=colors2, edgecolor='black', width=0.5)
ax2.set_title('$R^2$ Score (Higher is Better)')
ax2.set_ylabel('$R^2$ Value')
ax2.set_ylim(0, 0.6)  # Dynamic headroom for labels
ax2.grid(axis='y', linestyle='--', alpha=0.5)
# Add values on top of R2 bars (rounded to 3 decimal places)
ax2.bar_label(bars2, fmt='%.3f', padding=3)

plt.suptitle('Model Performance Comparison (Diabetes Dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()