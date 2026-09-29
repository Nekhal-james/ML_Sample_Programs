import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data (Fallback to synthetic if missing)
try:
    df = pd.read_excel("Online Retail.xlsx")
except FileNotFoundError:
    np.random.seed(42)
    df = pd.DataFrame({
        'InvoiceNo': np.random.randint(500000, 505000, 1000),
        'StockCode': np.random.randint(10000, 90000, 1000),
        'Quantity': np.random.randint(1, 20, 1000),
        'UnitPrice': np.random.uniform(1.0, 50.0, 1000).round(2),
        'InvoiceDate': pd.date_range('2025-01-01', periods=1000, freq='h'),
        'CustomerID': np.random.randint(10000, 10500, 1000)
    })

# 2. Preprocessing & Feature Engineering
df = df.dropna(subset=["CustomerID"]).query("Quantity > 0 and UnitPrice > 0").copy()
df["CustomerID"] = df["CustomerID"].astype(int)
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
customer_data = df.groupby("CustomerID").agg(
    TotalSpending=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfInvoices=("InvoiceNo", "nunique"),
    UniqueProducts=("StockCode", "nunique"),
    AverageTransactionValue=("TotalAmount", "mean"),
    Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days)
).reset_index()

customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpending"], q=3, labels=["Low Value", "Medium Value", "High Value"]
)

# --- Deliverable 1: Preprocessed customer-level dataset ---
print("--- PREPROCESSED CUSTOMER-LEVEL DATASET (HEAD) ---")
print(customer_data.head(), "\n")

# --- Deliverable 2: Customer segment distribution ---
print("--- CUSTOMER SEGMENT DISTRIBUTION ---")
print(customer_data["Segment"].value_counts(), "\n")

# 3. Model Training
features = ["TotalQuantity", "NumberOfInvoices", "UniqueProducts", "AverageTransactionValue", "Recency"]
X_train, X_test, y_train, y_test = train_test_split(
    customer_data[features], customer_data["Segment"], test_size=0.2, random_state=42, stratify=customer_data["Segment"]
)

dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42).fit(X_train, y_train)

# 4. Evaluation & Results
y_pred = dt.predict(X_test)

# --- Deliverable 3: Classification accuracy ---
print(f"--- CLASSIFICATION ACCURACY ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# --- Deliverable 4: Classification report ---
print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred), "\n")

# --- Deliverable 5: Feature importance values ---
importance = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False)
print("--- FEATURE IMPORTANCE VALUES ---")
print(importance.to_string(), "\n")
print(f"DECISION TREE RULES:\n{export_text(dt, feature_names=features)}")

# --- Deliverable 6 & 7: Decision Tree Visualization & Feature Importance Bar Chart ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Decision Tree Plot
plt.figure(figsize=(18, 10))
plot_tree(
    dt,
    feature_names=features,
    class_names=dt.classes_.astype(str),
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Customer Segmentation (ID3)", fontsize=16, fontweight='bold', pad=15)
plt.show()

# Figure 2: Feature Importance Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
importance_sorted = importance.sort_values(ascending=True)
bars = ax.barh(importance_sorted.index, importance_sorted.values, color='skyblue', edgecolor='navy')
ax.set_title("Feature Importance", fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Features", fontsize=12)

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
            va='center', ha='left', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()