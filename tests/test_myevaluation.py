from mysklearn.myevaluation import confusion_matrix, accuracy_score
from mysklearn.mynaviebayes import MyNaiveBayesClassifier


def test_confusion_matrix_and_accuracy():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "A", "B"]
    cm, labels = confusion_matrix(y_true, y_pred)
    # labels sorted: ["A","B"]
    assert labels == ["A", "B"]
    assert cm == [[1, 1], [1, 1]]
    assert accuracy_score(y_true, y_pred) == 0.5


def test_naive_bayes_simple_separation():
    X = [[0.0], [0.1], [1.0], [1.1]]
    y = ["L", "L", "R", "R"]

    nb = MyNaiveBayesClassifier()
    nb.fit(X, y)

    preds = nb.predict([[0.05], [1.05]])
    assert preds == ["L", "R"]
