# myrandomforest.py
# Random Forest (CPSC 322 compatible)

import random
from collections import Counter
from mysklearn.mydecisiontree import MyDecisionTreeClassifier
from mysklearn.myutils import majority_vote


class MyRandomForestClassifier:
    def __init__(self, n_estimators=20, max_features=2, top_k=7):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.top_k = top_k
        self.trees = []
        self.feature_importances_ = None

    # -------------------------------------
    # Bootstrap sample
    # -------------------------------------
    def bootstrap(self, X, y):
        n = len(X)
        idx = [random.randrange(n) for _ in range(n)]
        oob = [i for i in range(n) if i not in idx]

        X_boot = [X[i] for i in idx]
        y_boot = [y[i] for i in idx]
        X_oob = [X[i] for i in oob]
        y_oob = [y[i] for i in oob]

        return X_boot, y_boot, X_oob, y_oob

    # -------------------------------------
    # Fit
    # -------------------------------------
    def fit(self, X, y):
        trees_with_scores = []

        n_features = len(X[0])

        for _ in range(self.n_estimators):
            X_boot, y_boot, X_oob, y_oob = self.bootstrap(X, y)

            # Random feature subset
            self.features = random.sample(range(n_features), self.max_features)

            X_boot_sub = [[row[f] for f in self.features] for row in X_boot]
            X_oob_sub = [[row[f] for f in self.features] for row in X_oob]

            dt = MyDecisionTreeClassifier()
            dt.fit(X_boot_sub, y_boot)

            # OOB accuracy
            preds = dt.predict(X_oob_sub)
            acc = sum(1 for i in range(len(preds)) if preds[i] == y_oob[i]) / (len(preds) + 1e-9)

            trees_with_scores.append((acc, dt, self.features))

        # Select top-K trees
        trees_with_scores.sort(key=lambda x: x[0], reverse=True)
        selected = trees_with_scores[:self.top_k]

        self.trees = [(dt, feats) for (_, dt, feats) in selected]

    # -------------------------------------
    # Predict
    # -------------------------------------
    def predict(self, X):
        all_preds = []

        for row in X:
            votes = []
            for dt, feats in self.trees:
                row_sub = [row[f] for f in feats]
                votes.append(dt.predict([row_sub])[0])
            pred = majority_vote(votes)
            all_preds.append(pred)

        return all_preds
