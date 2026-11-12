import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv("../datasets/pima_indians_diabetes.csv")

df = df.dropna()
print(df.shape)

X = df.drop(["Outcome"], axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# without feature scaling
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_unscaled = accuracy_score(y_test, y_pred)
f1_unscaled = f1_score(y_test, y_pred)
precision_unscaled = precision_score(y_test, y_pred)
recall_unscaled = recall_score(y_test, y_pred)

print("\n-------Without Scaling---------")
print("Accuracy: ", accuracy_unscaled)
print("Precision: ", precision_unscaled)
print("Recall: ", recall_unscaled)
print("F1 Score: ", f1_unscaled)


# With feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
y_pred_scaled = model.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
f1_scaled = f1_score(y_test, y_pred_scaled)
precision_scaled = precision_score(y_test, y_pred_scaled)
recall_scaled = recall_score(y_test, y_pred_scaled)

print("\n-------With Scaling---------")
print("Accuracy: ", accuracy_scaled)
print("Precision: ", precision_scaled)
print("Recall: ", recall_scaled)
print("F1 Score: ", f1_scaled)


# Visualization
metrices = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrices_unscaled = [accuracy_unscaled, precision_unscaled, recall_unscaled, f1_unscaled]
metrices_scaled = [accuracy_scaled, precision_scaled, recall_scaled, f1_scaled]

x = np.arange(len(metrices))
width = 0.3
plt.bar(x - width/2, metrices_unscaled, width, label="Unscaled")
plt.bar(x + width/2, metrices_scaled, width, label="Scaled")
plt.xticks(x, metrices)
plt.ylabel("Scores")
plt.title("Logistic Regression with Scaling & Without Scaling")
plt.legend()
plt.show()

