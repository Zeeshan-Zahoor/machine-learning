import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

digits = load_digits()

X = digits.data

print("X shape: ", X.shape)

scaler = StandardScaler()

X = scaler.fit_transform(X)

k_values = [3, 5, 7]
inertias = []
silhouettes = []
for value in k_values:
    kmeans = KMeans(
        n_clusters=value,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, kmeans_labels)
    inertia = kmeans.inertia_

    silhouettes.append(silhouette)
    inertias.append(inertia)

    print(f"Silhouette with {value} clusters: ", silhouette)
    print(f"Inertia with {value} clusters: ", inertia)
    print()

# visualization
plt.subplot(1, 2, 1)
plt.bar(k_values, inertias)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("K-Means Inertia")

plt.subplot(1, 2, 2)
plt.bar(k_values, silhouettes)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Score")

plt.show()