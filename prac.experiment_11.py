import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

fashion_mnist = fetch_openml(
    name="Fashion-MNIST",
    version=1,
    as_frame=False,
    parser="liac-arff"
)

X = fashion_mnist.data
y = fashion_mnist.target.astype(int)

print("X Shape: ", X.shape)
print("Y Shape: ", y.shape)
print("classes: ", np.unique(y))

X, _, y, _ = train_test_split(
    X, y, train_size=5000,
    random_state=42,
    stratify=y
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

accuracies = []
for kernel in ["linear", "poly", "rbf"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Accuracy ({kernel}): ", accuracy)



# Visualization
plt.bar(["linear", "poly", "rbf"], accuracies, width=0.5)
plt.xlabel("Kernel Types")
plt.ylabel("Accuracy")
plt.title("SVM kernel comparision")
plt.legend()

plt.show()