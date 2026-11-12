import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

df = pd.read_csv("../datasets/auto_mpg.csv")
df = df.dropna()

print("Shape: ", df.shape)

X = df[["displacement"]].values
y = df["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

# Linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_lin = model.predict(X_test_scaled)

mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)


print("-------------Linear Regression-------------")
print("MSE: ", mse_lin)
print("R2 Score: ", r2_lin)

# Polynomial regression
degrees = [2, 3, 4]
poly_results = []
print("\n-------------Polynomial Regression----------------")
for degree in degrees: 
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # predict
    y_pred_poly = poly_model.predict(X_test_poly)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)

    poly_results.append(y_pred_poly)

    print("Degree: ", degree)
    print("MSE: ", mse_poly)
    print("R2 Score: ", r2_poly)
    print()

# visualization
idx = np.argsort(X_test_scaled[:, 0])
plt.scatter(X_train_scaled, y_train, s=10)
plt.plot(X_test_scaled[idx], y_pred_lin[idx], label="Linear Regression")

degree = 2
for result in poly_results:
    plt.plot(X_test_scaled[idx], result[idx], label=f"Degree: {degree}")
    degree = degree + 1

plt.legend()
plt.show()