import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../datasets/winequality-red.csv", sep=";")


df = df.dropna()
print("Shape: ", df.shape)
print("Columns: ", df.columns)

X = df.drop("quality", axis=1).values
y = df["quality"].values

print("Classes: ", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

architectures = [(50,), (50, 50), (50, 50, 50)] # single, two, and three hidden layers with 50 neurons each
accuracies = []

for layers in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Architecture {layers}: Accuracy = {accuracy}")


# Visualization
plt.bar(["1 Hidden Layer", "2 Hidden Layers", "3 Hidden Layers"], accuracies, width=0.5)
plt.xlabel("MLP Architecture")
plt.ylabel("Accuracy")
plt.title("MLP Architecture Comparison")
plt.show()