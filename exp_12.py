import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Fashion MNIST dataset from OpenML (replaces TensorFlow dependency)
print("Loading Fashion MNIST dataset from OpenML...")
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X = fashion_mnist.data
y = fashion_mnist.target.astype(int)

# Select a subset of the dataset to reduce SVM training time
# OpenML Fashion MNIST keeps the first 60,000 as train and the last 10,000 as test
X_train = X[:10000]
y_train = y[:10000]
X_test = X[60000:62000]
y_test = y[60000:62000]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Note: fetch_openml automatically provides data as flat 784-dimensional vectors
# Normalize the Pixel Values (Range 0 to 1)
X_train = X_train / 255.0
X_test = X_test / 255.0
print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())

# Implement Linear SVM
start_time = time.time()
linear_svm = SVC(kernel="linear", C=1.0)
linear_svm.fit(X_train, y_train)
linear_time = time.time() - start_time
y_pred_linear = linear_svm.predict(X_test)

linear_accuracy = accuracy_score(y_test, y_pred_linear)
linear_precision = precision_score(y_test, y_pred_linear, average="macro")
linear_recall = recall_score(y_test, y_pred_linear, average="macro")
linear_f1 = f1_score(y_test, y_pred_linear, average="macro")

print("\nLinear SVM")
print("Accuracy:", linear_accuracy)
print("Precision:", linear_precision)
print("Recall:", linear_recall)
print("F1-score:", linear_f1)
print("Training Time:", linear_time)

# Implement Polynomial SVM
start_time = time.time()
poly_svm = SVC(kernel="poly", degree=3, C=1.0, gamma="scale")
poly_svm.fit(X_train, y_train)
poly_time = time.time() - start_time
y_pred_poly = poly_svm.predict(X_test)

poly_accuracy = accuracy_score(y_test, y_pred_poly)
poly_precision = precision_score(y_test, y_pred_poly, average="macro")
poly_recall = recall_score(y_test, y_pred_poly, average="macro")
poly_f1 = f1_score(y_test, y_pred_poly, average="macro")

print("\nPolynomial SVM")
print("Accuracy:", poly_accuracy)
print("Precision:", poly_precision)
print("Recall:", poly_recall)
print("F1-score:", poly_f1)
print("Training Time:", poly_time)

# Implement RBF SVM
start_time = time.time()
rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_svm.fit(X_train, y_train)
rbf_time = time.time() - start_time
y_pred_rbf = rbf_svm.predict(X_test)

rbf_accuracy = accuracy_score(y_test, y_pred_rbf)
rbf_precision = precision_score(y_test, y_pred_rbf, average="macro")
rbf_recall = recall_score(y_test, y_pred_rbf, average="macro")
rbf_f1 = f1_score(y_test, y_pred_rbf, average="macro")

print("\nRBF SVM")
print("Accuracy:", rbf_accuracy)
print("Precision:", rbf_precision)
print("Recall:", rbf_recall)
print("F1-score:", rbf_f1)
print("Training Time:", rbf_time)

# Compare the Performance
results = pd.DataFrame({
    "Kernel": ["Linear", "Polynomial", "RBF"],
    "Accuracy": [linear_accuracy, poly_accuracy, rbf_accuracy],
    "Precision": [linear_precision, poly_precision, rbf_precision],
    "Recall": [linear_recall, poly_recall, rbf_recall],
    "F1-score": [linear_f1, poly_f1, rbf_f1],
    "Training Time": [linear_time, poly_time, rbf_time]
})

print("\nPerformance Comparison Summary Table:")
print(results)

# Display the Best Performing Kernel
best_kernel = results.loc[results["Accuracy"].idxmax(), "Kernel"]
best_accuracy = results["Accuracy"].max()
print("\nBest Performing Kernel:", best_kernel)
print("Best Accuracy:", best_accuracy)