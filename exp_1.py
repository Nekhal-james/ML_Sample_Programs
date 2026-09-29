import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
data = housing.frame
X = data[['AveRooms']].values
y = data['MedHouseVal'].values

np.random.seed(42)

shuffled_indices = np.random.permutation(len(X))
test_set_size = int(len(X) * 0.2)  

test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

def gradient_descent(X, y, alpha=0.01, iterations=5000):
    m = len(y)
    theta = np.zeros(X.shape[1]) # Initialize weights to 0
    
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        
    return theta

theta_gd = gradient_descent(X_train_bias, y_train, alpha=0.001, iterations=5000)
y_pred_gd = X_test_bias.dot(theta_gd)

def normal_equation(X, y):
    # theta = (X^T * X)^(-1) * X^T * y
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

theta_ne = normal_equation(X_train_bias, y_train)
y_pred_ne = X_test_bias.dot(theta_ne)

def evaluate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, r2

mse_gd, r2_gd = evaluate_metrics(y_test, y_pred_gd)
mse_ne, r2_ne = evaluate_metrics(y_test, y_pred_ne)

# Print Results
print("\nGRADIENT DESCENT RESULTS")
print(f"Intercept: {theta_gd[0]:.4f}, Slope: {theta_gd[1]:.4f}")
print(f"MSE: {mse_gd:.4f}, R² Score: {r2_gd:.4f}\n")

print("\nNORMAL EQUATION RESULTS")
print(f"Intercept: {theta_ne[0]:.4f}, Slope: {theta_ne[1]:.4f}")
print(f"MSE: {mse_ne:.4f}, R² Score: {r2_ne:.4f}\n")
# ----------------------------------------------------
# 5. Visualization with Dynamic Metrics in Titles
# ----------------------------------------------------
plt.figure(figsize=(14, 5))
sort_idx = np.argsort(X_train.flatten())

# Subplot 1: Gradient Descent Fit
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.2, label='Data')
plt.plot(X_train[sort_idx], X_train_bias[sort_idx].dot(theta_gd), color='red', linewidth=2, label='GD Line')
# Dynamic title string showing metrics
plt.title(f"Gradient Descent Fit\nMSE: {mse_gd:.4f} | R² Score: {r2_gd:.4f}")
plt.xlabel("AveBedrms")
plt.ylabel("MedHouseVal")
plt.legend()

# Subplot 2: Normal Equation Fit
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, color='blue', alpha=0.2, label='Data')
plt.plot(X_train[sort_idx], X_train_bias[sort_idx].dot(theta_ne), color='green', linewidth=2, label='Normal Eq Line')
# Dynamic title string showing metrics
plt.title(f"Normal Equation Fit\nMSE: {mse_ne:.4f} | R² Score: {r2_ne:.4f}")
plt.xlabel("AveBedrms")
plt.ylabel("MedHouseVal")
plt.legend()

plt.tight_layout()
plt.show()