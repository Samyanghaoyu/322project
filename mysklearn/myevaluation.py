# myevaluation.py
# Custom evaluation utilities for CPSC 322 (No sklearn allowed)

import random
from collections import Counter


# =========================================================
# 1. Train-Test Split
# =========================================================
def train_test_split(X, y, test_size=0.2, random_state=None):
    """Simple implementation of train_test_split without sklearn."""
    
    if random_state is not None:
        random.seed(random_state)

    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    test_n = int(n * test_size)
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test



# =========================================================
# 2. K-Fold Split
# =========================================================
def kfold_split(X, n_splits=5, random_state=None):
    """Return list of (train_indices, test_indices) for normal k-fold."""
    
    if random_state is not None:
        random.seed(random_state)

    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // n_splits
    folds = []

    for k in range(n_splits):
        start = k * fold_size
        end = start + fold_size if k < n_splits - 1 else n

        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, test_idx))

    return folds



# =========================================================
# 3. Stratified K-Fold Split
# =========================================================
def stratified_kfold_split(X, y, n_splits=5, random_state=None):
    """Stratified k-fold split to preserve class ratios."""
    
    if random_state is not None:
        random.seed(random_state)

    # separate indices by class
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    # shuffle each class bucket
    for label in class_indices:
        random.shuffle(class_indices[label])

    # initialize folds
    folds = [[] for _ in range(n_splits)]

    # distribute indices round-robin
    for label, idx_list in class_indices.items():
        for i, idx in enumerate(idx_list):
            folds[i % n_splits].append(idx)

    # convert each fold to (train, test)
    result = []
    all_indices = list(range(len(X)))

    for i in range(n_splits):
        test_idx = sorted(folds[i])
        train_idx = sorted(list(set(all_indices) - set(test_idx)))
        result.append((train_idx, test_idx))

    return result



# =========================================================
# 4. Bootstrap Sampling
# =========================================================
def bootstrap_sample(X, y, n_samples=None, random_state=None):
    """Return bootstrap sample (sampling with replacement)."""
    
    if random_state is not None:
        random.seed(random_state)

    if n_samples is None:
        n_samples = len(X)

    indices = [random.randrange(0, len(X)) for _ in range(n_samples)]

    X_boot = [X[i] for i in indices]
    y_boot = [y[i] for i in indices]

    return X_boot, y_boot



# =========================================================
# 5. Confusion Matrix
# =========================================================
def confusion_matrix(y_true, y_pred, labels=None):
    """Generate confusion matrix with rows = true labels, columns = predicted."""
    
    if labels is None:
        labels = sorted(list(set(y_true)))

    matrix = [[0 for _ in labels] for _ in labels]

    label_to_idx = {label: i for i, label in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        i = label_to_idx[t]
        j = label_to_idx[p]
        matrix[i][j] += 1

    return matrix, labels



# =========================================================
# 6. Accuracy Score
# =========================================================
def accuracy_score(y_true, y_pred):
    """Compute accuracy as #correct / total."""
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)
