import pandas as pd
import numpy as np
import random
from typing import Callable, Optional, Union


class MyLogReg:
    eps = 1e-15

    def __init__(self, n_iter: int = 10, learning_rate: int | float | Callable = 0.1, weights: Optional[tuple] = None,
                 metric: Optional[str] = None, reg: Optional[str] = None, l1_coef: float = 0, l2_coef: float = 0,
                 sgd_sample: Optional[int | float] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metric_value = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        return "MyLineReg class: " + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])

    @staticmethod
    def __sigmoid(z: np.array) -> np.array:
        return 1 / (1 + np.exp(-z))

    def __write_log(self, iter: int, learning_rate: float, logloss: float, verbose: int) -> None:
        log_str = f"{iter} | learning ratr: {learning_rate} | logloss: {logloss}"
        if self.metric: log_str += f" | {self.metric}: {self.metric_value}"
        print(log_str)

    def __calc_metric(self, X: np.array, y: np.array) -> float:
        scores = self.__sigmoid(X @ self.weights)
        y_pred = scores > 0.5

        TP = (y * y_pred).sum()
        TN = ( (y == 0) * (y_pred == 0) ).sum()
        FP = ( (y == 0) * y_pred ).sum()
        FN = ( y * (y_pred == 0) ).sum()

        if self.metric == "accuracy":
            return (TP + TN) / (TP + TN + FP + FN)

        elif self.metric == "precision":
            return TP / (TP + FP)

        elif self.metric == "recall":
            return TP / (TP + FN)

        elif self.metric == "f1":
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            return 2 * (precision * recall) / (precision + recall)

        elif self.metric == "roc_auc":
            table = sorted( tuple(zip(scores, y)), reverse=True )
            summa = 0
            for indx, (score, target) in enumerate(table):
                if not target:
                    summa += sum( tuple(row[0] > score and row[1] for row in table[:indx]) )
                    summa += sum( tuple(row[0] == score and row[1] for row in table[:indx]) ) / 2
            P, N = y.sum(), y.size - y.sum() # positive, negatove
            return round(summa * 1 / (P * N), 10)

    def __get_reg_members(self) -> tuple:
        loss_reg = grad_reg = 0

        if self.reg == "l1":
            loss_reg = self.l1_coef * np.linalg.norm(self.weights, ord=1)
            grad_reg = self.l1_coef * np.sign(self.weights)

        elif self.metric == "l2":
            loss_reg = self.l2_coef * np.linalg.norm(self.weights, ord=2) ** 2
            grad_reg = self.l2_coef * 2 * self.weights

        elif self.metric == "elasticnet":
            loss_reg = self.l1_coef * np.linalg.norm(self.weights, ord=1) + self.l2_coef * np.linalg.norm(self.weights, ord=2) ** 2
            grad_reg = self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights

        return loss_reg, grad_reg

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False) -> None:
        random.seed(self.random_state)

        X, y = np.array(X), np.array(y)
        N, D = X.shape
        X = np.hstack([np.ones(N).reshape(-1, 1), X])
        self.weights = np.ones(D + 1)

        for iter in range(1, self.n_iter + 1):
            # prediction
            logits = X @ self.weights
            y_pred = p = self.__sigmoid(logits)

            # logloss
            logloss = - (1 / N) * np.sum( y * np.log(y_pred + self.eps) + (1 - y) * np.log(1 - y_pred + self.eps) )
            logloss += self.__get_reg_members()[0]

            # gradient
            if self.sgd_sample:
                batch_size = self.sgd_sample if type(self.sgd_sample) == int else round(N * self.sgd_sample)
                sample_rows_idx = random.sample(range(N), batch_size)
                mini_X, mini_y = X[sample_rows_idx], y[sample_rows_idx]
                mini_y_pred = self.__sigmoid(mini_X @ self.weights)
                grad = (1 / batch_size) * (mini_X.T @ (mini_y_pred - mini_y)) + self.__get_reg_members()[1]
            else:
                grad = (1 / N) * (X.T @ (y_pred - y)) + self.__get_reg_members()[1]

            # updating
            learning_rate = self.learning_rate(iter) if callable(self.learning_rate) else self.learning_rate
            self.weights -= learning_rate * grad

            # metric
            if self.metric and self.metric != "roc_auc": # roc-auc is too long
                self.metric_value = self.__calc_metric(X, y)

            # logging
            if verbose: self.__write_log(iter, self.learning_rate, logloss, verbose)

        self.metric_value = self.__calc_metric(X, y)

    def get_coef(self) -> np.array:
        return self.weights[1:]

    def get_best_score(self) -> float:
        return self.metric_value

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = np.array(X)
        N, D = X.shape
        X = np.hstack( [np.ones(N).reshape(-1, 1), X] )
        return self.__sigmoid(X @ self.weights)

    def predict(self, X: pd.DataFrame) -> np.array:
        p = self.predict_proba(X)
        return p > 0.5