import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("../datasets/iris.csv")
print(df.head())

# Features and target
X = df[["sepal.length", "sepal.width"]].values
y = (df["variety"] == "Setosa").astype(int).values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training shape: ", X_train.shape)
print("Testing shape: ", X_test.shape)

# Train Linear SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
# print("Accuracy: ", accuracy_score(y_test, y_pred))


# Split original, unscaled data
x_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM without scalling
model_raw = SVC(kernel="linear")
model_raw.fit(x_train_raw, y_train_raw)

# Predict and compare accuracy
y_pred_raw = model_raw.predict(X_test_raw)

print("Accuracy without scaling:", accuracy_score(y_test_raw, y_pred_raw))
print("Accuracy with scaling:", accuracy_score(y_test, y_pred))


# Visualization
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    facecolors="none", edgecolors="red",
    label="Support Vectors"
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM")
plt.legend()



#Hyperplane and margins
w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

plt.plot(x, -(w[0] * x + b) / w[1], "b-")
plt.plot(x, -(w[0] * x + b - 1) / w[1], "g--")
plt.plot(x, -(w[0] * x + b + 1) / w[1], "g--")


plt.show()