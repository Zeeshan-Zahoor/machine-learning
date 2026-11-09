import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt

df = pd.read_csv("../datasets/20_newsgroups.csv")

df = df.dropna()

print("Shape: ", df.shape) 



X = df["text"]
y = df["target_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples: ", len(X_train))
print("Testing samples: ", len(X_test))

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)

accuracy = []
f1 = []


model1 = MultinomialNB()
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

accuracy_1 = accuracy_score(y_test, y_pred1)
f1_score_1 = f1_score(y_test, y_pred1, average="weighted")
accuracy.append(accuracy_1)
f1.append(f1_score_1)

print("\n---------- Multinomial Naive Bayes -----------")
print("Accuracy: ", accuracy_1)
print("F1 Score: ", f1_score_1)



model2 = BernoulliNB()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

accuracy_2 = accuracy_score(y_test, y_pred2)
f1_score_2 = f1_score(y_test, y_pred2, average="weighted")
accuracy.append(accuracy_2)
f1.append(f1_score_2)

print("\n----------- Bernoulli Naive Bayes -------------")
print("Accuracy: ", accuracy_2)
print("F1 Score: ", f1_score_2)


# visualization
models = ["Multinomial NB", "Bernoulli NB"]

x = np.arange(len(models))
width = 0.3

plt.bar(x - width/2, accuracy, width=width, label="Accuracy")
plt.bar(x + width/2, f1, width=width, label="F1 Score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Naive Bayes Model Comparision")
plt.legend()
plt.show()