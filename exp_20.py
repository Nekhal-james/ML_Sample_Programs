import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Step 1 & 2: Load the Boston Housing Dataset
print("--- Loading Dataset ---")
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Step 3: Select Input and Target
# Using 'rm' (average number of rooms) as input and 'medv' (median value) as target
X = df[["rm"]]
y = df["medv"]

# Step 4: Split the Dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# Step 5: Define Polynomial Degrees
degrees = [1, 2, 3, 4, 5, 7, 10, 15]
train_errors = []
validation_errors = []

# Step 6: Train Polynomial Regression Models
print("\n--- Training Polynomial Models ---")
for degree in degrees:
    # Create a pipeline that scales data, adds polynomial features, and trains a linear model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regressor", LinearRegression())
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Calculate Mean Squared Error
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    
    # Store errors
    train_errors.append(train_mse)
    validation_errors.append(val_mse)

# Step 7: Create a Comparison Table
results = pd.DataFrame({
    "Polynomial Degree": degrees,
    "Training MSE": train_errors,
    "Validation MSE": validation_errors
})
print("\nPerformance Comparison:")
print(results.to_string(index=False))

# Step 8: Identify the Degree with Minimum Validation Error
best_index = np.argmin(validation_errors)
best_degree = degrees[best_index]
best_validation_error = validation_errors[best_index]

print("\n--- Final Results ---")
print(f"Best polynomial degree: {best_degree}")
print(f"Minimum validation MSE: {best_validation_error:.4f}")

# Step 9: Plot Training and Validation Errors
plt.figure(figsize=(9, 6))
plt.plot(degrees, train_errors, marker="o", label="Training Error (Bias)", color="blue")
plt.plot(degrees, validation_errors, marker="o", label="Validation Error (Variance)", color="orange")

# Formatting the plot
plt.xlabel("Polynomial Degree (Model Complexity)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Bias-Variance Tradeoff: Polynomial Regression on Boston Housing")
plt.xticks(degrees)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()