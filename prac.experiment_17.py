import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = sns.load_dataset("titanic")
df = df[["pclass", "sex", "age", "sibsp", "fare", "survived"]]
df = df.dropna()
df["sex"] = df["sex"].map({"male": 0, "female": 1})

print(df.head())
print("Shape: ", df.shape)
print("Columns: ", df.columns)

X = df.drop("survived", axis=1).values
y = df["survived"].values


print("X Shape:", X.shape)
print("Y Shape:", y.shape)
print("Classes:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Bagging 
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

print("\nBagging Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_bag))
print("Precision:", precision_score(y_test, y_pred_bag))
print("Recall:", recall_score(y_test, y_pred_bag))
print("F1 Score:", f1_score(y_test, y_pred_bag))

# Boosting
boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

boosting.fit(X_train, y_train)
y_pred_boost = boosting.predict(X_test)
print("\nBoosting Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_boost))
print("Precision:", precision_score(y_test, y_pred_boost))
print("Recall:", recall_score(y_test, y_pred_boost))
print("F1 Score:", f1_score(y_test, y_pred_boost))