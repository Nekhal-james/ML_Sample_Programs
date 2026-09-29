# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Libraries imported successfully!\n")

# Step 2: Load the Wine Quality Dataset
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
try:
    data = pd.read_csv(url, sep=";")
    print("Dataset shape:", data.shape)
    print("\nFirst five rows:")
    print(data.head())
except Exception as e:
    print("Error loading dataset. Please check your internet connection or URL.", e)

# Step 3: Separate Features and Target
X = data.drop("quality", axis=1)
y = data["quality"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
print("\nUnique Quality values:", sorted(y.unique()))

# Step 4: Check for Missing Values
print("\nMissing values per column:")
print(data.isnull().sum())

# Step 5: Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 6: Standardize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining data after scaling:")
print("Mean:", round(X_train.mean(), 4))
print("Standard deviation:", round(X_train.std(), 4))

# Step 7: Define Different MLP Architectures
architectures = {
    "1 Hidden Layer - 16 Neurons": (16,),
    "1 Hidden Layer - 32 Neurons": (32,),
    "2 Hidden Layers - 32,16": (32, 16),
    "2 Hidden Layers - 64,32": (64, 32),
    "3 Hidden Layers - 64,32,16": (64, 32, 16)
}

print("\nDefined Architectures:")
for name, architecture in architectures.items():
    print(f"- {name} : {architecture}")

# Step 8 & 9: Train, Evaluate, and Store Results for each MLP Architecture
results = []

for name, architecture in architectures.items():
    print(f"\nTraining architecture: {name}...")
    start_time = time.time()
    
    # Initialize MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
    
    # Train the model
    mlp.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predict on test set
    y_pred = mlp.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    results.append([
        name,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(training_time, 4)
    ])
    
    print(f"  -> Accuracy: {accuracy:.4f}")
    print(f"  -> Precision: {precision:.4f}")
    print(f"  -> Recall: {recall:.4f}")
    print(f"  -> F1-score: {f1:.4f}")
    print(f"  -> Training Time: {training_time:.4f} seconds")

# Step 10: Create the Performance Comparison Table
results_df = pd.DataFrame(
    results,
    columns=[
        "Architecture",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Training Time (s)"
    ]
)

print("\n" + "="*50)
print("PERFORMANCE COMPARISON TABLE")
print("="*50)
print(results_df.to_string(index=False))

# Step 11: Display the Best Performing Architecture
best_index = results_df["Accuracy"].idxmax()
best_architecture = results_df.loc[best_index, "Architecture"]
best_accuracy = results_df.loc[best_index, "Accuracy"]

print("\n" + "="*50)
print("FINAL CONCLUSION")
print("="*50)
print(f"Best Performing Architecture : {best_architecture}")
print(f"Best Accuracy Achieved       : {best_accuracy * 100:.2f}%")