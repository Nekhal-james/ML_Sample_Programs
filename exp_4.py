import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)

models = {
    'MLE': LogisticRegression(penalty=None, max_iter=10000, random_state=42).fit(X_tr, y_tr),
    'MAP (L1)': LogisticRegression(penalty='l1', solver='liblinear', random_state=42).fit(X_tr, y_tr),
    'MAP (L2)': LogisticRegression(penalty='l2', random_state=42).fit(X_tr, y_tr)
}

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
colors = ['red', 'blue', 'green']
accs = [accuracy_score(y_te, m.predict(X_te)) * 100 for m in models.values()]

axes[0, 0].bar(models.keys(), accs, color=colors)
axes[0, 0].set(ylabel='Accuracy (%)', ylim=(80, 100), title='1. Accuracy Comparison')
axes[0, 0].grid(axis='y', linestyle='--')

for ax, (name, m), c in zip(axes.flat[1:], models.items(), colors):
    ax.plot(m.coef_[0], color=c, marker='o', ms=4)
    ax.axhline(0, color='black', ls='--')
    ax.set(title=f'Weights: {name}', xlabel='Feature', ylabel='Weight')
    ax.grid(True)

plt.tight_layout(); plt.show()