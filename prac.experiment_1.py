import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("../datasets/california_housing.csv")

print("Dataset shape: ", df.shape)

X = df[["MedInc"]].values
y = df["MedHouseVal"].values

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
)

# preprocess
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

# using gradient descent
theta_0 = 0
theta_1 = 0
learning_rate = 0.1
epoches = 1000

for epoch in range(epoches):

    # predict
    prediction = theta_0 + theta_1 * X_train_scaled[:, 0]

    # error
    error = prediction - y_train

    # update parameters
    dtheta_0 = np.mean(error)
    dtheta_1 = np.mean(error * X_train_scaled[:, 0])

    theta_0 = theta_0 - learning_rate * dtheta_0
    theta_1 = theta_1 - learning_rate * dtheta_1

# make prediction
y_pred_gd = theta_0 + theta_1 * X_test_scaled[:, 0]
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

print("\n----------------------------------------")
print("Gradient Descent Results")
print("----------------------------------------")
print("Theta0:", theta_0)
print("Theta1:", theta_1)
print("Test MSE:", mse_gd)
print("Test R2:", r2_gd)


# Normal equation
X_train_ne = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_ne = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# predict
theta_ne = np.linalg.inv(X_train_ne.T @ X_train_ne) @ X_train_ne.T @ y_train

# prediction
y_pred_ne = X_test_ne @ theta_ne
mse_ne = mean_squared_error(y_test, y_pred_ne)
r2_ne = r2_score(y_test, y_pred_ne)

print("\n----------------------------------------")
print("Normal Equation Results")
print("----------------------------------------")

print("Theta0:", theta_ne[0])
print("Theta1:", theta_ne[1])
print("Test MSE:", mse_ne)
print("Test R2:", r2_ne)


# Visualization
plt.scatter(X_train_scaled, y_train, label="Actual Data")
plt.plot(X_test_scaled, y_pred_gd, label="Gradient Descent")
plt.plot(X_test_scaled, y_pred_ne, label="Equation")

plt.xlabel("MedInc (Scaled)")
plt.ylabel("MedHouseVal")
plt.title("Linear Regression")

plt.legend()
plt.show()