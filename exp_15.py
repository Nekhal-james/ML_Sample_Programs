# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("Libraries imported successfully!\n")

# Step 2: Load the Fashion MNIST Dataset via Scikit-Learn
print("Loading Fashion MNIST dataset (this may take a moment)...")
# Using data_id 40996 bypasses naming lookup issues on OpenML
fashion_mnist = fetch_openml(data_id=40996, as_frame=False, parser='auto')
X, y = fashion_mnist.data, fashion_mnist.target.astype(int)

# Standard Fashion MNIST split: first 60,000 for training, last 10,000 for testing
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Number of classes:", len(np.unique(y_train)))

# Step 3: Normalize the Pixel Values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\nMinimum pixel value:", X_train.min())
print("Maximum pixel value:", X_test.max())

# Step 4 & 5: Define Hyperparameter Combinations to Test
# Note: In scikit-learn's MLPClassifier, 
# - learning rate is configured via 'learning_rate_init'
# - batch size is configured via 'batch_size'
# - maximum epochs are configured via 'max_iter'
experiments = [
    {"learning_rate": 0.001, "batch_size": 32,  "epochs": 10},
    {"learning_rate": 0.001, "batch_size": 128, "epochs": 10},
    {"learning_rate": 0.001, "batch_size": 256, "epochs": 10},
    {"learning_rate": 0.01,  "batch_size": 128, "epochs": 10},
    {"learning_rate": 0.0001,"batch_size": 128, "epochs": 10},
    {"learning_rate": 0.001, "batch_size": 128, "epochs": 5},
    {"learning_rate": 0.001, "batch_size": 128, "epochs": 20}
]

print("\nHyperparameter tuning experiments defined.")

# Step 6: Experiment with Different Hyperparameters
results = []

for i, exp in enumerate(experiments, 1):
    lr = exp["learning_rate"]
    bs = exp["batch_size"]
    ep = exp["epochs"]
    
    print(f"\n--- Experiment {i} ---")
    print(f"Learning Rate: {lr} | Batch Size: {bs} | Epochs: {ep}")
    
    # Initialize MLP Classifier with fixed architecture (128 hidden neurons) and chosen hyperparameters
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        batch_size=bs,
        max_iter=ep,
        random_state=42,
        verbose=False
    )
    
    # Train the model
    start_time = time.time()
    mlp.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate on test set
    y_pred = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    results.append([lr, bs, ep, round(test_accuracy, 4), round(training_time, 2)])
    
    print(f"  -> Test Accuracy: {test_accuracy:.4f}")
    print(f"  -> Training Time: {training_time:.2f} seconds")

# Step 7: Create the Performance Comparison Table
results_df = pd.DataFrame(
    results,
    columns=["Learning Rate", "Batch Size", "Epochs", "Accuracy", "Training Time (s)"]
)

print("\n" + "="*60)
print("PERFORMANCE COMPARISON TABLE")
print("="*60)
print(results_df.to_string(index=False))

# Step 8: Display the Best Hyperparameter Combination
best_index = results_df["Accuracy"].idxmax()
best_result = results_df.loc[best_index]

print("\n" + "="*60)
print("FINAL CONCLUSION / BEST COMBINATION")
print("="*60)
print(f"Learning Rate : {best_result['Learning Rate']}")
print(f"Batch Size    : {int(best_result['Batch Size'])}")
print(f"Epochs        : {int(best_result['Epochs'])}")
print(f"Test Accuracy : {best_result['Accuracy'] * 100:.2f}%")
print(f"Training Time : {best_result['Training Time (s)']} seconds")