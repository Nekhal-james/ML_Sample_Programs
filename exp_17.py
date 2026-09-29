import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1 & 2: Load the Digits Dataset
print("--- Loading Dataset ---")
digits = load_digits()
X = digits.data
y = digits.target
print("Dataset shape:", X.shape)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# Step 3: Preprocess the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled data shape:", X_scaled.shape)

# Step 4: Apply K-Means for Different Values of K
k_values = [2, 3, 4, 5, 6, 8, 10, 12]
inertia_values = []
silhouette_values = []

print("\n--- Training K-Means Models ---")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate and store metrics
    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels))

# Step 5: Create and Print a Comparison Table
results = pd.DataFrame({
    "Number of Clusters (K)": k_values,
    "Inertia": inertia_values,
    "Silhouette Score": silhouette_values
})
print("\nComparison Table:")
print(results)

# Step 6: Identify the Best K Based on Silhouette Score
best_index = np.argmax(silhouette_values)
best_k = k_values[best_index]
best_score = silhouette_values[best_index]

print("\n--- Final Results ---")
print(f"Best K based on silhouette score: {best_k}")
print(f"Best silhouette score: {best_score:.4f}")

# Step 7: Plot Everything in One Window
# Create a figure with 1 row and 3 columns of plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: Sample Digit
axes[0].imshow(digits.images[0], cmap="gray")
axes[0].set_title("Sample Digit")
axes[0].axis("off")

# Subplot 2: K vs Inertia
axes[1].plot(k_values, inertia_values, marker="o", color="blue")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Inertia")
axes[1].set_title("K-Means: K vs Inertia")
axes[1].set_xticks(k_values)
axes[1].grid(True)

# Subplot 3: K vs Silhouette Score
axes[2].plot(k_values, silhouette_values, marker="o", color="orange")
axes[2].set_xlabel("Number of Clusters (K)")
axes[2].set_ylabel("Silhouette Score")
axes[2].set_title("K-Means: K vs Silhouette Score")
axes[2].set_xticks(k_values)
axes[2].grid(True)

# Adjust layout to prevent overlap and display the window
plt.tight_layout()
plt.show()