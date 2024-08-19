import pandas as pd
import numpy as np
from numpy.linalg import norm

class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.train = None
        self.targets = None
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"MyKNNReg class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X, y = X.to_numpy(), y.to_numpy()
        self.train = X
        self.targets = y
        self.train_size = X.shape

    def __get_distance(self, vector1: np.array, vector2: np.array) -> float:
        if self.metric == "euclidean":
            return norm(vector1 - vector2, ord=2)
        elif self.metric == "chebyshev":
            return norm(vector1 - vector2, ord=np.inf)
        elif self.metric == "manhattan":
            return norm(vector1 - vector2, ord=1)
        elif self.metric == "cosine":
            return 1 - ( (vector1 @ vector2) / (norm(vector1) * (norm(vector2))) )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.to_numpy()
        y = []

        for sample in X:
            distances = np.array([self.__get_distance(sample, train_sample) for train_sample in self.train])

            indxs = np.argsort(distances)[:self.k]
            distances = distances[indxs]
            targets = self.targets[indxs]

            if self.weight == "uniform":
                weights = np.array( [1 / self.k for i in range(self.k)] )
            elif self.weight == "rank":
                denom = sum( 1 / (i+1) for i in range(self.k) )
                weights = np.array( [(1 / (i+1)) / denom for i in range(self.k)] )
            elif self.weight == "distance":
                denom = sum(1 / distances[i] for i in range(self.k))
                weights = np.array([(1 / distances[i]) / denom for i in range(self.k)])

            y.append(weights @ targets)

        return pd.Series(y)