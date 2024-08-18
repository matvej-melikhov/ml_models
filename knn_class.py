import pandas as pd
import numpy as np
from numpy.linalg import norm


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.train = None
        self.targets = None
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"MyKNNClf class: k={self.k}"

    def __get_distance(self, vector1: np.array, vector2: np.array) -> float: # norm
        if self.metric == "euclidean":
            return norm(vector1 - vector2, ord=2)
        elif self.metric == "chebyshev":
            return norm(vector1 - vector2, ord=np.inf)
        elif self.metric == "manhattan":
            return norm(vector1 - vector2, ord=1)
        elif self.metric == "cosine":
            return 1 - ( (vector1 @ vector2) / (norm(vector1) * (norm(vector2))) )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X, y = X.to_numpy(), y.to_numpy()
        self.train = X
        self.targets = y
        self.train_size = X.shape

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X = X.to_numpy()
        y = []

        for sample in X:
            distances = np.array([self.__get_distance(sample, train_sample) for train_sample in self.train])

            indxs = np.argsort(distances)[:self.k]
            distances = distances[indxs]
            targets = self.targets[indxs]

            if self.weight == "uniform":
                weights = np.array([1] * len(targets))
            elif self.weight == "rank":
                weights = np.array([1 / (i + 1) for i in range(len(targets))])
            elif self.weight == "distance":
                weights = np.array([1 / distances[i] for i in range(len(targets))])

            w1 = np.array([weights[i] for i in range(len(weights)) if targets[i] == 1])
            y.append( w1.sum() / weights.sum() )

        return pd.Series(y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probs = self.predict_proba(X)
        return probs >= 0.5