from mysklearn.myrandomforest import MyRandomForestClassifier
from mysklearn.mydecisiontree import MyDecisionTreeClassifier


def test_bootstrap_and_stratified_split_sizes():
    # small dataset with imbalance
    X = [[0], [1], [2], [3], [4], [5]]
    y = ["A", "A", "A", "B", "B", "B"]

    rf = MyRandomForestClassifier(random_state=0)
    X_train, X_test, y_train, y_test = rf.stratified_split(X, y, test_ratio=1/3)

    # Expect roughly 1/3 test (rounding handled in impl)
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(X)
    # Stratified: both labels should appear in train
    assert set(y_train) == {"A", "B"}

    X_boot, y_boot, X_oob, y_oob = rf.bootstrap(X_train, y_train)
    assert len(X_boot) == len(X_train)
    # OOB can be empty, but all boot samples come from train
    assert all(row in X_train for row in X_boot)


def test_decision_tree_random_feature_subset_root_choice():
    # With max_features=1 and fixed seed, root should select feature index 1
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = ["L", "R", "L", "R"]  # depends on second feature

    dt = MyDecisionTreeClassifier(max_features=1, random_state=0)
    dt.fit(X, y)
    assert dt.tree.attribute == 1  # selected feature subset is {1}


def test_random_forest_fit_predict_simple_and():
    # AND function
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = ["N", "N", "N", "Y"]

    rf = MyRandomForestClassifier(
        n_estimators=10, top_k=5, max_features=1, random_state=42
    )
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds == ["N", "N", "N", "Y"]


def test_random_forest_majority_vote_behavior():
    # Predictions should always be one of the seen labels and match input length
    X = [[i] for i in range(6)]
    y = ["A", "A", "A", "A", "B", "B"]

    rf = MyRandomForestClassifier(
        n_estimators=5, top_k=3, max_features=1, random_state=1
    )
    rf.fit(X, y)
    preds = rf.predict([[10], [0]])
    assert len(preds) == 2
    assert set(preds).issubset({"A", "B"})
