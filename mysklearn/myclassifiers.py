# myclassifiers.py
# A collection of custom classifiers for CPSC 322
# Includes: KNN + unified interface

import math
from collections import Counter


# =====================================================
#           K-Nearest Neighbors Classifier
# =====================================================

class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    # ------------------------
    # Fit
    # ------------------------
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # ------------------------
    # Distance function
    # ------------------------
    def distance(self, x1, x2):
        # Euclidean distance for numeric features
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

    # ------------------------
    # Predict one
    # ------------------------
    def predict_one(self, row):
        # Compute all distances
        dists = []
        for train_row, label in zip(self.X_train, self.y_train):
            d = self.distance(row, train_row)
            dists.append((d, label))

        # Sort by distance
        dists.sort(key=lambda x: x[0])

        # Select K nearest
        k_labels = [label for (_, label) in dists[:self.k]]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    # ------------------------
    # Predict many
    # ------------------------
    def predict(self, X):
        return [self.predict_one(row) for row in X]


# =====================================================
#      Unified interface for consistent experiments
# =====================================================

class MyClassifier:
    """Wraps different custom classifiers under a unified interface."""

    def __init__(self, classifier_type="knn", **kwargs):
        """
        classifier_type options:
        - "knn"
        - "dt"
        - "nb"
        - "rf"
        """
        if classifier_type == "knn":
            from mysklearn.myclassifiers import MyKNeighborsClassifier
            self.model = MyKNeighborsClassifier(**kwargs)

        elif classifier_type == "dt":
            from mysklearn.mydecisiontree import MyDecisionTreeClassifier
            self.model = MyDecisionTreeClassifier()

        elif classifier_type == "nb":
            from mysklearn.mynaviebayes import MyNaiveBayesClassifier
            self.model = MyNaiveBayesClassifier()

        elif classifier_type == "rf":
            from mysklearn.myrandomforest import MyRandomForestClassifier
            self.model = MyRandomForestClassifier(**kwargs)

        else:
            raise ValueError("Invalid classifier_type")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class MyDummyClassifier:
    """Always predict the majority class."""
    def __init__(self):
        self.majority = None

    def fit(self, X, y):
        from collections import Counter
        self.majority = Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return [self.majority for _ in X]
