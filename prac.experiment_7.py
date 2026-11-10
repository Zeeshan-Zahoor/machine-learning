import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("../datasets/20_newsgroups.csv")

df = df.dropna()

X = df["text"]

X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)
print("Dataset Shape: ", df.shape)
print("Training Samples: ", len(X_train))
print("Testing Samples: ", len(X_test))

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)

word_counts = np.asarray(X_train.sum(axis=0)).flatten()

print("Vocabulary Size: ", len(word_counts))
print("Total Work Count: ", word_counts.sum())

# MLE estimation
total_words =  word_counts.sum()

mle_probs = word_counts / total_words

print("\n--- MLE Estimation ---")
print("Sum of probabilities: ", mle_probs.sum())
print("First 10 word probabilities: ", mle_probs[:10])

# MAP estimation using Dirichlet priors

V = len(word_counts)

# Prior 1: alpha = 1.5
alpha1 = 1.5
map_probs1 = (word_counts + alpha1 - 1) / (total_words + V * (alpha1 - 1))

# Prior 2: alpha = 2
alpha2 = 2
map_probs2 = (word_counts + alpha2 - 1) / (total_words + V * (alpha2 - 1))

print("\n--- MAP Estimation ---")
print("Alpha = 1.5, Sum:", map_probs1.sum())
print("Alpha = 2, Sum:", map_probs2.sum())



plt.figure(figsize=(12, 5))

x = np.arange(20)

plt.plot(x, mle_probs[:20], marker="o", label="MLE")
plt.plot(x, map_probs1[:20], marker="s", label="MAP (Alpha=1.5)")
plt.plot(x, map_probs2[:20], marker="^", label="MAP (Alpha=2)")

plt.xlabel("Word Index")
plt.ylabel("Probability")
plt.title("MLE vs MAP Probability Estimation")
plt.legend()
plt.show()