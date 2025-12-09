from mysklearn.myclassifiers import MyKNeighborsClassifier


def test_knn_two_class_prediction():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = ["A", "A", "B", "B"]

    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    preds = knn.predict([[0, 0], [1, 1], [0.2, 0.8]])
    assert preds == ["A", "B", "A"]
