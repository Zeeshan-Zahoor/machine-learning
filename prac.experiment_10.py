import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("../datasets/iris.csv")
df = df.dropna()
print(df.head())
print("Shape: ", df.shape)

X = df[["sepal.width", "petal.width"]].values
y = (df["variety"] == "Setosa").astype(int).values

X_train, X_test,y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

plt.scatter(X_train[:, 0], X_train[:, 1], label="Training Data", c=y_train)
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], facecolors="None", edgecolors="red", label="Support Vectors",)

# Hyperplace and Margins
w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)

y = -(w[0] * x + b) / w[1]

# margins
y_margin1 = -(w[0] * x + b - 1) / w[1]
y_margin2 = -(w[0] * x + b + 1) / w[1]

plt.plot(x, y, label="Hyperplane")
plt.plot(x, y_margin1, "k--", label="Margins")
plt.plot(x, y_margin2, 'k--')

plt.title("Linear SVM")

plt.legend()
plt.show()
