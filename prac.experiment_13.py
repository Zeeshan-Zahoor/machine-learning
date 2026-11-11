import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml(
    name="mnist_784",
    version=1,
    as_frame=False,
    parser="liac-arff"
)

X = mnist.data
y = mnist.target.astype(int)

print("X Shape: ", X.shape)
print("Y Shape: ", y.shape)
print("Classes: ", np.unique(y))

X, _, y, _ = train_test_split(
    X, y, train_size=6000,
    random_state=42,
    stratify=y
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training shape: ", X_train.shape)
print("Testing shape: ", X_test.shape)

activations = ["logistic", "relu", "tanh"]

accuracies = []
training_times = []
iterations = []

for activation in activations:
    model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation=activation,
            max_iter=200,
            random_state=42
        )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    training_time = end - start

    accuracies.append(accuracy)
    training_times.append(training_time)
    iterations.append(model.n_iter_)

    print(f"\nActivation: {activation}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Iterations: {model.n_iter_}")


# visualization 
plt.subplot(1, 3, 1)
plt.bar(activations, accuracies, width=0.5)
plt.xlabel("Activations")
plt.ylabel("Accuracy")
plt.title("Accuracy")

plt.subplot(1, 3, 2)
plt.bar(activations, training_times, width=0.5)
plt.xlabel("Activations")
plt.ylabel("Training Time")
plt.title("Training Time")

plt.subplot(1, 3, 3)
plt.bar(activations, iterations, width=0.5)
plt.xlabel("Activations")
plt.ylabel("Iterations")
plt.title("Iterations")

plt.tight_layout()

plt.show()