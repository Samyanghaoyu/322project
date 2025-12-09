# myrandomforest.py
# Random Forest (CPSC 322 compatible)

import random
from mysklearn.mydecisiontree import MyDecisionTreeClassifier
from mysklearn.myutils import majority_vote


class MyRandomForestClassifier:
    def __init__(self, n_estimators=20, max_features=2, top_k=7, random_state=None):
        """
        n_estimators (N): total trees to train
        top_k (M): keep the best-performing trees on validation
        max_features (F): number of candidate attributes per split
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.top_k = top_k
        self.trees = []
        self.feature_importances_ = None
        self.rng = random.Random(random_state)

    # -------------------------------------
    # Bootstrap sample
    # -------------------------------------
    def bootstrap(self, X, y):
        n = len(X)
        idx = [self.rng.randrange(n) for _ in range(n)]
        oob = [i for i in range(n) if i not in idx]

        X_boot = [X[i] for i in idx]
        y_boot = [y[i] for i in idx]
        X_oob = [X[i] for i in oob]
        y_oob = [y[i] for i in oob]

        return X_boot, y_boot, X_oob, y_oob

    # -------------------------------------
    # Stratified split (1/3 test, 2/3 remainder)
    # -------------------------------------
    def stratified_split(self, X, y, test_ratio=1/3):
        buckets = {}
        for i, label in enumerate(y):
            buckets.setdefault(label, []).append(i)

        train_idx, test_idx = [], []
        for idxs in buckets.values():
            self.rng.shuffle(idxs)
            split = max(1, round(len(idxs) * test_ratio))
            test_idx.extend(idxs[:split])
            train_idx.extend(idxs[split:])

        # fallback if nothing in train
        if len(train_idx) == 0:
            train_idx, test_idx = test_idx, []

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        return X_train, X_test, y_train, y_test

    # -------------------------------------
    # Fit
    # -------------------------------------
    def fit(self, X, y):
        trees_with_scores = []

        n_features = len(X[0])

        for _ in range(self.n_estimators):
            X_boot, y_boot, X_oob, y_oob = self.bootstrap(X, y)

            dt = MyDecisionTreeClassifier(max_features=self.max_features, random_state=self.rng.randrange(1_000_000))
            dt.fit(X_boot, y_boot)

            # Validation accuracy (OOB)
            if len(X_oob) == 0:
                acc = 0
            else:
                preds = dt.predict(X_oob)
                acc = sum(1 for i in range(len(preds)) if preds[i] == y_oob[i]) / len(preds)

            trees_with_scores.append((acc, dt))

        # Select top-K trees
        trees_with_scores.sort(key=lambda x: x[0], reverse=True)
        selected = trees_with_scores[: self.top_k]

        self.trees = [dt for (acc, dt) in selected]

    # -------------------------------------
    # Predict
    # -------------------------------------
    def predict(self, X):
        all_preds = []

        for row in X:
            votes = []
            for dt in self.trees:
                votes.append(dt.predict([row])[0])
            pred = majority_vote(votes)
            all_preds.append(pred)

        return all_preds
