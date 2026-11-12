from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

fashion_mnist = fetch_openml(
     name="Fashion-MNIST",
     version=1,
     as_frame=False,
     parser="liac-arff"
)

X = fashion_mnist.data
y = fashion_mnist.target.astype(int)

X = X / 255

X, _, y, _ = train_test_split(
     X, y, train_size=12000, random_state=42, stratify=y
)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42, stratify=y
)

k_values = [3, 5, 7]
accuracies = []

initial_accuracy = 0
for k in k_values:
    model = KNeighborsClassifier(
                n_neighbors=k
            )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > initial_accuracy:
        initial_accuracy = accuracy
        best_pred = y_pred

    accuracies.append(accuracy)

    print("K values: ", k)
    print("Accuracy: ", accuracy)
    print()

cm = confusion_matrix(y_test, best_pred)

# visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matix Against the Best K")
plt.show()

plt.bar(k_values, accuracies)
plt.xlabel("K Values")
plt.ylabel("Accuracy")

plt.show()