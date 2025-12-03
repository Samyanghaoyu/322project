# mydecisiontree.py
# Pure Python Decision Tree Classifier for CPSC 322 Final Project
# Compatible with PA-style data and no sklearn needed

import math
from collections import Counter


class DecisionNode:
    """Node of a decision tree."""
    def __init__(self, attribute=None, threshold=None, branches=None, label=None):
        # Internal node
        self.attribute = attribute      # column index
        self.threshold = threshold      # numeric split
        self.branches = branches or {}  # dict: key -> child node

        # Leaf node
        self.label = label              # class label at leaf

    def is_leaf(self):
        return self.label is not None


class MyDecisionTreeClassifier:
    """Simple decision tree classifier supporting numeric features."""

    def __init__(self):
        self.tree = None

    # ============================================================
    # Main public API
    # ============================================================
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    # ============================================================
    # Entropy & Information Gain
    # ============================================================
    def entropy(self, labels):
        counter = Counter(labels)
        total = len(labels)
        ent = 0
        for count in counter.values():
            p = count / total
            ent -= p * math.log2(p)
        return ent

    def info_gain_numeric(self, X_column, y, threshold):
        """Compute information gain for a numeric split."""
        left_y = [y[i] for i in range(len(y)) if X_column[i] <= threshold]
        right_y = [y[i] for i in range(len(y)) if X_column[i] > threshold]

        if len(left_y) == 0 or len(right_y) == 0:
            return 0  # invalid split

        H_before = self.entropy(y)
        H_left = self.entropy(left_y)
        H_right = self.entropy(right_y)
        w_left = len(left_y) / len(y)
        w_right = len(right_y) / len(y)

        H_after = w_left * H_left + w_right * H_right
        return H_before - H_after

    # ============================================================
    # Tree building
    # ============================================================
    def best_numeric_split(self, X_column, y):
        """Find best threshold for numeric attribute."""
        values = sorted(set(X_column))
        best_gain = 0
        best_threshold = None

        # try midpoints
        for i in range(len(values) - 1):
            thr = (values[i] + values[i + 1]) / 2
            gain = self.info_gain_numeric(X_column, y, thr)
            if gain > best_gain:
                best_gain = gain
                best_threshold = thr

        return best_gain, best_threshold

    def choose_best_attribute(self, X, y):
        """Return (best_attribute_index, best_threshold)."""
        n_features = len(X[0])
        best_attr = None
        best_threshold = None
        best_gain = 0

        for col in range(n_features):
            X_column = [row[col] for row in X]
            gain, thr = self.best_numeric_split(X_column, y)

            if gain > best_gain:
                best_gain = gain
                best_attr = col
                best_threshold = thr

        return best_attr, best_threshold

    def build_tree(self, X, y):
        # If all labels are the same → leaf
        if len(set(y)) == 1:
            return DecisionNode(label=y[0])

        # If no features left
        if len(X[0]) == 0:
            return DecisionNode(label=Counter(y).most_common(1)[0][0])

        # Choose best attribute
        attr, threshold = self.choose_best_attribute(X, y)

        # If no useful split → leaf
        if attr is None or threshold is None:
            return DecisionNode(label=Counter(y).most_common(1)[0][0])

        # Split data
        left_X, left_y = [], []
        right_X, right_y = [], []

        for i, row in enumerate(X):
            if row[attr] <= threshold:
                left_X.append(row)
                left_y.append(y[i])
            else:
                right_X.append(row)
                right_y.append(y[i])

        # If split fails → leaf
        if len(left_y) == 0 or len(right_y) == 0:
            return DecisionNode(label=Counter(y).most_common(1)[0][0])

        # Build subtrees
        left_child = self.build_tree(left_X, left_y)
        right_child = self.build_tree(right_X, right_y)

        return DecisionNode(
            attribute=attr,
            threshold=threshold,
            branches={"left": left_child, "right": right_child}
        )

    # ============================================================
    # Prediction
    # ============================================================
    def predict_one(self, x):
        node = self.tree

        while not node.is_leaf():
            attr = node.attribute
            thr = node.threshold

            if x[attr] <= thr:
                node = node.branches["left"]
            else:
                node = node.branches["right"]

        return node.label
