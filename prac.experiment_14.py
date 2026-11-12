import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("../datasets/Mall_Customers.csv")
df = df.dropna()

print(df.head())
print(df.shape)

X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10,
)

kmeans_labels = kmeans.fit_predict(X)

print("K-Means Silhoutte Score: ", silhouette_score(X, kmeans_labels))
print("K-Means Inertia: ", kmeans.inertia_)


# Hierarchical Clustering
for link in ["single", "complete"]:

    hierarchical = AgglomerativeClustering(
        n_clusters=5,
        linkage=link
    )

    labels = hierarchical.fit_predict(X)

    if link == "single":
        labels_single = labels
    else:
        labels_complete = labels

    print(
        f"{link} Linkage Silhouette Score:",
        silhouette_score(X, labels)
    )

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("K-Means")

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_single)
plt.title("Single Linkage")

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels_complete)
plt.title("Complete Linkage")

plt.show()