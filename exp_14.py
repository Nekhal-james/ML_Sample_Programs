# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("Libraries imported successfully!\n")

# Step 2: Load the MNIST Dataset via Scikit-Learn
print("Loading MNIST dataset (this may take a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Standard MNIST split: first 60,000 for training, last 10,000 for testing
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

# Create a small validation split (10% of training data) to track validation accuracy per epoch
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# Function to train and evaluate using scikit-learn's MLPClassifier iteratively (Epoch simulation)
def train_mlp_with_epochs(activation_name, sklearn_activation, epochs=10):
    print(f"\n--- Training {activation_name} Network ---")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation=sklearn_activation,
        solver="adam",
        random_state=42
    )
    
    classes = np.unique(y_train)
    train_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
    # Simulate epochs using partial_fit
    for epoch in range(epochs):
        if epoch == 0:
            mlp.partial_fit(X_tr, y_tr, classes=classes)
        else:
            mlp.partial_fit(X_tr, y_tr)
            
        # Capture loss
        train_losses.append(mlp.loss_)
        
        # Capture validation accuracy
        y_val_pred = mlp.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {mlp.loss_:.4f} - Val Accuracy: {val_acc:.4f}")
        
    training_time = time.time() - start_time
    
    # Evaluate on test set
    y_pred = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    history = {
        "loss": train_losses,
        "val_accuracy": val_accuracies
    }
    
    return test_accuracy, training_time, history

# ==========================================
# Step 5, 6 & 7: Train Sigmoid, ReLU, and Tanh Networks
# ==========================================
# Note: scikit-learn uses 'logistic' for the sigmoid activation function
sigmoid_acc, sigmoid_time, hist_sigmoid = train_mlp_with_epochs("Sigmoid", "logistic", epochs=10)
relu_acc, relu_time, hist_relu = train_mlp_with_epochs("ReLU", "relu", epochs=10)
tanh_acc, tanh_time, hist_tanh = train_mlp_with_epochs("Tanh", "tanh", epochs=10)

# ==========================================
# Step 8: Create the Performance Comparison Table
# ==========================================
results_data = [
    ["Sigmoid", sigmoid_acc, sigmoid_time],
    ["ReLU", relu_acc, relu_time],
    ["Tanh", tanh_acc, tanh_time]
]

results_df = pd.DataFrame(
    results_data,
    columns=["Activation Function", "Test Accuracy", "Training Time (s)"]
)

print("\n" + "="*50)
print("PERFORMANCE COMPARISON TABLE")
print("="*50)
print(results_df.to_string(index=False))

# ==========================================
# Step 9: Plot Training Loss Comparison
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(hist_sigmoid["loss"], label="Sigmoid", linewidth=2)
plt.plot(hist_relu["loss"], label="ReLU", linewidth=2)
plt.plot(hist_tanh["loss"], label="Tanh", linewidth=2)
plt.xlabel("Epoch", fontweight='bold')
plt.ylabel("Training Loss", fontweight='bold')
plt.title("Training Loss Comparison Across Activation Functions", fontweight='bold', pad=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

# ==========================================
# Step 10: Plot Validation Accuracy Comparison
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(hist_sigmoid["val_accuracy"], label="Sigmoid", linewidth=2)
plt.plot(hist_relu["val_accuracy"], label="ReLU", linewidth=2)
plt.plot(hist_tanh["val_accuracy"], label="Tanh", linewidth=2)
plt.xlabel("Epoch", fontweight='bold')
plt.ylabel("Validation Accuracy", fontweight='bold')
plt.title("Validation Accuracy Comparison Across Activation Functions", fontweight='bold', pad=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

# ==========================================
# Step 11: Display the Best Performing Activation Function
# ==========================================
best_index = results_df["Test Accuracy"].idxmax()
best_activation = results_df.loc[best_index, "Activation Function"]
best_accuracy = results_df.loc[best_index, "Test Accuracy"]

print("\n" + "="*50)
print("FINAL CONCLUSION")
print("="*50)
print(f"Best Activation Function : {best_activation}")
print(f"Best Test Accuracy       : {best_accuracy * 100:.2f}%")