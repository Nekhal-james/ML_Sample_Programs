import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

iris = load_iris()

X = iris.data[:,:2]
y = np.where(iris.target == 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

w = model.coef_[0]
b = model.intercept_[0]

print("Weight:", w)
print("Bias:", b)
print("Support Vectors:", len(model.support_vectors_))

xx = np.linspace(X_train[:,0].min()-1, X_train[:,0].max()+1, 500)
boundary = -(w[0]*xx + b) / w[1]
margin1 = -(w[0]*xx + b - 1) / w[1]
margin2 = -(w[0]*xx + b + 1) / w[1]

plt.figure(figsize=(9,6))

plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1],
            label="Setosa")

plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1],
            label="Non-Setosa")

plt.plot(xx, boundary, label="Decision Boundary")
plt.plot(xx, margin1, "--", label="Margin")
plt.plot(xx, margin2, "--")

sv = model.support_vectors_
plt.scatter(sv[:,0], sv[:,1], s=120,
            facecolors="none", edgecolors="black",
            label="Support Vectors")

plt.xlabel("Standardized Petal Length")
plt.ylabel("Standardized Petal Width")
plt.title("Linear SVM")
plt.legend()
plt.grid(True)
plt.show()

print("Margin Width:", 2 / np.linalg.norm(w))