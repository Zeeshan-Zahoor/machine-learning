import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


df = pd.read_csv("../datasets/diabetes_prediction_dataset.csv")

df = df.dropna()
print(df.shape)

X = df.drop(["diabetes", "gender", "smoking_history"], axis=1).values
y = df["diabetes"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear regression 
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("--------- Linear Regression ----------")
print("MSE: ", mse_linear)
print("R2 Score: ", r2_linear)


alphas = np.logspace(-3, 3, 20)

# Ridge + Cross validation
ridge_model = RidgeCV(
    alphas=alphas,
    cv=5
)

ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
best_alphs_ridge = ridge_model.alpha_
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\n--------- Rigde Regression ----------")
print("MSE: ", mse_ridge)
print("R2 Score: ", r2_ridge)
print("Best Alpha: ", best_alphs_ridge)

# Lasso + Cross validation
lasso_model = LassoCV(
    alphas=alphas,
    cv=5
)

lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
best_alphs_lasso = lasso_model.alpha_
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\n--------- Lasso Regression ----------")
print("MSE: ", mse_lasso)
print("R2 Score: ", r2_lasso)
print("Best Alpha: ", best_alphs_lasso)


# Visualization
models = ["Linear", "Ridge", "Lasso"]
mse = [mse_linear, mse_ridge, mse_lasso]

plt.bar(models, mse)
plt.xlabel("Regression Models")
plt.ylabel("Mean Squared Error")
plt.title("Linear vs Ridge vs Lasso Regression")

plt.show()