import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("../datasets/iris.csv")
df = df.dropna()

X = df.drop("variety", axis=1).values
y = df["variety"].values

print("Training shape: ", X.shape)
print("Testing shape: ", y.shape)
print("Classes: ", np.unique(y))


# Bootstrapping
accuracies = []
f1_scores = []

for i in range(10):

    indices = np.random.choice(len(X), len(X), replace=True)

    X_boot = X[indices]
    y_boot = y[indices]

    # out of bag
    oob = np.setdiff1d(np.arange(len(X)), indices) 

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_boot, y_boot)

    y_pred = model.predict(X[oob])

    accuracies.append(accuracy_score(y[oob], y_pred))
    f1_scores.append(f1_score(y[oob], y_pred, average="weighted"))

print("Bootstrap Accuracy:", np.mean(accuracies))
print("Bootstrap F1 Score:", np.mean(f1_scores))


# K-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeClassifier(random_state=42)

results = cross_validate(
    model, X, y,
    cv=kfold,
    scoring=["accuracy", "f1_weighted"]
)

print("K-Fold Accuracy: ", np.mean(results["test_accuracy"]))
print("K-Fold F1 Score: ", np.mean(results["test_f1_weighted"]))
