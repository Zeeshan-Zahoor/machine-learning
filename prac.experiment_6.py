import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("../datasets/breast_cancer_wisconsin.csv")
df = df.dropna()

print("Shape: ", df.shape)

X = df.drop(df[["target"]], axis=1).values
y = df["target"]

print("X shape: ", X.shape)
print("y shape: ", y.shape)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Shape: ", X_train_scaled.shape)
print("Testing Shape: ", X_test_scaled.shape)

# Logistic Regression using MLE (No Regularization)
model_mle = LogisticRegression(penalty=None, max_iter=1000)
model_mle.fit(X_train_scaled, y_train)
y_pred_mle = model_mle.predict(X_test_scaled)

accuracy_mle = accuracy_score(y_test, y_pred_mle)
f1_mle = f1_score(y_test, y_pred_mle)

print("\n----- MLE Logistic Regression -----")
print("Accuracy: ", accuracy_mle)
print("F1 Score: ", f1_mle)


# MAP using L1 Regularization (Lasso)
model_l1 = LogisticRegression(
    penalty="l1", solver="liblinear", max_iter=1000
)

model_l1.fit(X_train_scaled, y_train)
y_pred_l1 = model_l1.predict(X_test_scaled)

# MAP using L2 Regularization (Ridge)
model_l2 = LogisticRegression(
    penalty='l2', solver="liblinear", max_iter=1000
)

model_l2.fit(X_train_scaled, y_train)
y_pred_l2 = model_l2.predict(X_test_scaled)

print("\n --- MAP L1 Regularization ---")
print("Accuracy: ", accuracy_score(y_test, y_pred_l1))
print("F1 Score: ", f1_score(y_test, y_pred_l1))

print("\n --- MAP L2 Regularization ---")
print("Accuracy: ", accuracy_score(y_test, y_pred_l2))
print("F1 Score: ", f1_score(y_test, y_pred_l2))


print("\n--- Parameter Comparison ---")

print("MLE Coefficients:", model_mle.coef_[0][:5])
print("L1 Coefficients:", model_l1.coef_[0][:5])
print("L2 Coefficients:", model_l2.coef_[0][:5])

# Visualization

plt.plot(model_mle.coef_[0], label="MLE")
plt.plot(model_l1.coef_[0], label="MAP L1")
plt.plot(model_l2.coef_[0], label="MAP L2")

plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Parameter Comparision: MLE vs MAP")
plt.legend()
plt.show()