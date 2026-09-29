import ssl, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

ssl._create_default_https_context = ssl._create_unverified_context

data = fetch_20newsgroups(subset='train', categories=['comp.graphics', 'sci.space'], remove=('headers', 'footers', 'quotes'))
vec = CountVectorizer(max_features=1000, stop_words='english')
counts = np.asarray(vec.fit_transform(data.data).sum(axis=0)).flatten()
N, V = counts.sum(), len(counts)

df = pd.DataFrame({'Word': vec.get_feature_names_out(), 'Count': counts, 'MLE': counts / N})
for a in [0.1, 2.0, 5.0]: 
    df[f'MAP {a}'] = (counts + a - 1) / (N + V * (a - 1))
top = df.sort_values(by='Count', ascending=False).head(5)

fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(5), 0.2
cols = df.columns[2:]  # MLE and MAP columns

for i, col in enumerate(cols):
    bars = ax.bar(x + i*w, top[col], w, color='lightgray', edgecolor='black')
    for b in bars:
        h = b.get_height()
        # Show name (e.g. MLE, MAP 0.1) and precise value
        ax.annotate(f'{col}\n{h:.4f}', (b.get_x() + w/2, h), 
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=6, rotation=90)

ax.set(xticks=x + 1.5*w, xticklabels=top['Word'], ylabel='Probability', ylim=(0, top['MLE'].max() * 1.5))
plt.tight_layout()
plt.show()