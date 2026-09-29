import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

# Step 1 & 2: Load the Iris Dataset
print("--- Loading Dataset ---")
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset shape:", X.shape)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", iris.target_names)

# Step 3: Create the Classification Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 4: Implement Bootstrapping
print("\n--- Running Bootstrapping ---")
rng = np.random.default_rng(42)
n_iterations = 30
bootstrap_results = []
n_samples = len(X)

for i in range(n_iterations):
    # Generate bootstrap sample (with replacement)
    train_indices = rng.choice(n_samples, size=n_samples, replace=True)
    
    # Identify out-of-bag (OOB) samples
    selected = np.zeros(n_samples, dtype=bool)
    selected[train_indices] = True
    test_indices = np.where(~selected)[0]
    
    # Train and evaluate only if OOB samples exist
    if len(test_indices) > 0:
        model.fit(X[train_indices], y[train_indices])
        y_pred = model.predict(X[test_indices])
        
        accuracy = accuracy_score(y[test_indices], y_pred)
        f1 = f1_score(y[test_indices], y_pred, average="macro")
        
        bootstrap_results.append([i + 1, accuracy, f1, len(test_indices)])

# Convert bootstrap results to a DataFrame
bootstrap_results_df = pd.DataFrame(
    bootstrap_results,
    columns=["Iteration", "Accuracy", "F1-Score", "OOB Samples"]
)

# Step 5: Calculate Bootstrap Performance
bootstrap_accuracy = bootstrap_results_df["Accuracy"].mean()
bootstrap_f1 = bootstrap_results_df["F1-Score"].mean()
print(f"Bootstrap Mean Accuracy: {bootstrap_accuracy:.4f}")
print(f"Bootstrap Mean F1-Score: {bootstrap_f1:.4f}")

# Step 6: Implement 5-Fold Cross-Validation
print("\n--- Running 5-Fold Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    model, X, y, cv=cv, scoring=["accuracy", "f1_macro"]
)

# Step 7: Calculate Cross-Validation Performance
cv_accuracy = cv_results["test_accuracy"].mean()
cv_f1 = cv_results["test_f1_macro"].mean()
print(f"Cross-Validation Mean Accuracy: {cv_accuracy:.4f}")
print(f"Cross-Validation Mean F1-Score: {cv_f1:.4f}")

# Step 8: Compare Both Resampling Methods
comparison = pd.DataFrame({
    "Method": ["Bootstrapping", "5-Fold Cross-Validation"],
    "Mean Accuracy": [bootstrap_accuracy, cv_accuracy],
    "Mean F1-Score": [bootstrap_f1, cv_f1]
})

print("\n--- Final Performance Comparison ---")
print(comparison.to_string(index=False))