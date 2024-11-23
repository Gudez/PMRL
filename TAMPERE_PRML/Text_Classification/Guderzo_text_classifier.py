from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer

# import the data
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

# a) run NN classifier for different sizes of vocabulary
# instead of using absolute "random" values, I would rather try to study
# the efficiency of the vocabulary size by giving a proportion of words to use

performance = {key: np.array([]) for key in ["english","nonstop"]}
features_proportions = np.arange(0.1, 1.1, 0.1)
y_train = data_train.target
y_test = data_test.target

print("Vectorization with stop word begins...")

# first I iterate through the different proportions with stop_word = "english"
stop_word = "english"
for proportion in features_proportions:

    max_features = int(proportion * len(data_train.data))
    vectorizer = CountVectorizer(stop_words=stop_word, max_features=max_features)

    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    performance[stop_word] = np.append(performance[stop_word], clf.score(X_test, y_test))

print("Vectorization without stop word begins...")

# same without stop word
stop_word = "nonstop"
for proportion in features_proportions:

    max_features = int(proportion * len(data_train.data))
    vectorizer = CountVectorizer(max_features=max_features)

    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    performance[stop_word] = np.append(performance[stop_word], clf.score(X_test, y_test))

# since with this loop I always put a "max_features", I'm adding as last
# value in the list the performance without max_features

## stop english
vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

# k = 3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

performance["english"] = np.append(performance["english"], clf.score(X_test, y_test))

## nonstop
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target

# k = 3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

performance["nonstop"] = np.append(performance["nonstop"], clf.score(X_test, y_test))

# print the result of the iteration
print(performance)

print("Plotting begins...")

# b) Plot the results
# I) full vocabulary accuracy with and without stop words
accuracy_with = performance["english"][10]
accuracy_without = performance["nonstop"][10]

plt.axhline(y=accuracy_with, color='red', linestyle='--', label='Full with stop words')
plt.axhline(y=accuracy_without, color='blue', linestyle='-', label='Full without stop words')

# II) Accuracy with limited vocabularity (w/ and w/o stop words)
plt.plot(features_proportions*len(data_train.data), performance["english"][0:10], color='green', marker='o', linestyle='-', label='With stop words')
plt.plot(features_proportions*len(data_train.data), performance["nonstop"][0:10], color='black', marker='o', linestyle='-', label='Without stop words')

print("TF-IDF begins...")

# c) TF-IDF weighting of the feature vectors
performance_TF = {key: np.array([]) for key in ["english","nonstop"]}
# features_proportions = np.arange(0.1, 1.1, 0.1)
tfidf_transformer = TfidfTransformer()
y_train = data_train.target
y_test = data_test.target

# first I iterate through the different proportions with stop_word = "english"

for proportion in features_proportions:
    stop_word = "english"
    max_features = int(proportion * len(data_train.data))
    vectorizer = CountVectorizer(stop_words=stop_word, max_features=max_features)

    X_train = tfidf_transformer.fit_transform(vectorizer.fit_transform(data_train.data))
    X_test = tfidf_transformer.fit_transform(vectorizer.fit_transform(data_test.data))

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train.toarray(), y_train)

    performance_TF[stop_word] = np.append(performance_TF[stop_word], clf.score(X_test.toarray(), y_test))

# same without stop word
for proportion in features_proportions:
    stop_word = "nonstop"
    max_features = int(proportion * len(data_train.data))
    vectorizer = CountVectorizer(max_features=max_features)

    X_train = tfidf_transformer.fit_transform(vectorizer.fit_transform(data_train.data))
    X_test = tfidf_transformer.fit_transform(vectorizer.fit_transform(data_test.data))

    # k = 3
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train.toarray(), y_train)

    performance_TF[stop_word] = np.append(performance_TF[stop_word], clf.score(X_test.toarray(), y_test))

plt.plot(features_proportions*len(data_train.data), performance_TF["english"], color='orange', marker='o', linestyle='-', label='TF with stop words')
plt.plot(features_proportions*len(data_train.data), performance_TF["nonstop"], color='purple', marker='o', linestyle='-', label='TF without stop words')

# Adding labels and title
plt.xlabel('Vocabulary size')
plt.ylabel('Classification accuracy')
plt.title('Accuracy with and without stop words for different sizes')
plt.legend()
plt.show()