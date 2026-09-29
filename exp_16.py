# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

print("Libraries imported successfully!\n")

# Step 2: Load the Mall Customers Dataset
# Using a reliable raw CSV link
url = "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/Mall_Customers.csv"

try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully!")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
except Exception as e:
    print("Error loading dataset. Please check your internet connection.", e)

# Step 3: Select Features
# Selecting Annual Income and Spending Score as the main numerical features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\nFeature preview:")
print(X.head())
print("\nMissing values per column:")
print(X.isnull().sum())

# Step 4: Standardize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nShape of scaled data:", X_scaled.shape)

# Step 5: Find a Suitable Number of Clusters Using Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores for different K values
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker="o", linestyle='-', color='b')
plt.xlabel("Number of Clusters (K)", fontweight='bold')
plt.ylabel("Silhouette Score", fontweight='bold')
plt.title("Silhouette Score for Different K Values", fontweight='bold', pad=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

# Step 6: Apply K-Means Clustering (Optimal K = 5)
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

print("\n" + "="*40)
print("K-MEANS CLUSTERING RESULTS")
print("="*40)
print(f"K-Means Inertia          : {kmeans_inertia:.4f}")
print(f"K-Means Silhouette Score : {kmeans_silhouette:.4f}")

# Step 7: Apply Agglomerative Hierarchical Clustering (Optimal K = 5, Ward linkage)
agglomerative = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)
agg_labels = agglomerative.fit_predict(X_scaled)
agg_silhouette = silhouette_score(X_scaled, agg_labels)

print("\n" + "="*40)
print("AGGLOMERATIVE CLUSTERING RESULTS")
print("="*40)
print(f"Agglomerative Silhouette Score : {agg_silhouette:.4f}")

# Step 8 & 9: Visualize K-Means and Agglomerative Clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# K-Means Scatter Plot
scatter1 = ax1.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=kmeans_labels,
    cmap='viridis',
    s=60,
    edgecolors='k',
    alpha=0.8
)
ax1.set_xlabel("Annual Income (k$)", fontweight='bold')
ax1.set_ylabel("Spending Score (1-100)", fontweight='bold')
ax1.set_title("K-Means Clustering (K=5)", fontweight='bold', pad=12)
ax1.grid(True, linestyle=":", alpha=0.6)

# Agglomerative Scatter Plot
scatter2 = ax2.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=agg_labels,
    cmap='plasma',
    s=60,
    edgecolors='k',
    alpha=0.8
)
ax2.set_xlabel("Annual Income (k$)", fontweight='bold')
ax2.set_ylabel("Spending Score (1-100)", fontweight='bold')
ax2.set_title("Agglomerative Hierarchical Clustering (K=5, Ward)", fontweight='bold', pad=12)
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# Step 10: Compare the Results in a Table
comparison = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Agglomerative"
    ],
    "Inertia": [
        round(kmeans_inertia, 4),
        np.nan  # Standard agglomerative clustering does not output inertia natively
    ],
    "Silhouette Score": [
        round(kmeans_silhouette, 4),
        round(agg_silhouette, 4)
    ]
})

print("\n" + "="*50)
print("PERFORMANCE COMPARISON TABLE")
print("="*50)
print(comparison.to_string(index=False))