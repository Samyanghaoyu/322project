from mysklearn.mydecisiontree import MyDecisionTreeClassifier


def test_decision_tree_predicts_and_function():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = ["N", "N", "N", "Y"]

    dt = MyDecisionTreeClassifier()
    dt.fit(X, y)

    preds = dt.predict(X)
    assert preds == y
