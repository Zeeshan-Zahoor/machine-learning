import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

df = pd.read_csv("datasets/iris.csv")
df = df.dropna()

print(df.head())
print("Shape: ", df.shape)

X = df.drop(df[["variety"]], axis=1).values
y = df["variety"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n --- Decision Tree Classifier ---")
print("Accuracy: ", accuracy)

importance = model.feature_importances_
plt.bar(df.columns[:-1], importance, width=0.5)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

plot_tree(model, filled=True)
plt.show()