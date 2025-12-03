# mynaviebayes.py
# Simple Gaussian Naive Bayes

import math
from collections import defaultdict

class MyNaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}

    def fit(self, X, y):
        self.classes = set(y)
        data = {c: [] for c in self.classes}

        for i, row in enumerate(X):
            data[y[i]].append(row)

        total = len(X)
        for c in self.classes:
            rows = data[c]
            self.prior[c] = len(rows) / total

            cols = list(zip(*rows))
            self.mean[c] = [sum(col)/len(col) for col in cols]
            self.var[c] = [sum((x - m)**2 for x in col)/len(col)
                           for col, m in zip(cols, self.mean[c])]

    def gaussian(self, x, mean, var):
        if var == 0:
            return 1e-9
        return math.exp(-(x - mean)**2 / (2*var)) / math.sqrt(2 * math.pi * var)

    def predict_one(self, row):
        post = {}
        for c in self.classes:
            prob = self.prior[c]
            for i, x in enumerate(row):
                prob *= self.gaussian(x, self.mean[c][i], self.var[c][i])
            post[c] = prob
        return max(post, key=post.get)

    def predict(self, X):
        return [self.predict_one(row) for row in X]
