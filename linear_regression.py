import random
import numpy as np
import pandas as pd
import math
from typing import Optional, Callable


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate: int | float | Callable = 0.1, metric: Optional[str] = None,
                 reg: Optional[str] = None, l1_coef: float = 0, l2_coef: float = 0,
                 sgd_sample: Optional[int | float] = None, random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        return "MyLineReg class: " + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])

    def __get_grad_reg(self) -> float:
        reg_grad = 0
        if self.reg == "l1":          reg_grad = self.l1_coef * np.sign(self.weights)
        if self.reg == "l2":          reg_grad = 2 * self.l2_coef * self.weights
        if self.reg == "elasticnet":  reg_grad = self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

        return reg_grad

    def __gradient_descent(self, X: np.array, y: np.array) -> float:
        N = X.shape[0]
        y_pred = X @ self.weights
        grad = (2 / N) * (X.T @ (y_pred - y))

        return grad + self.__get_grad_reg()

    def __stochastic_gradient_descent(self, X: np.array, y: np.array) -> float:
        N = X.shape[0]

        batch_size = N
        if type(self.sgd_sample) is int:
            batch_size = self.sgd_sample
        elif type(self.sgd_sample) is float:
            batch_size = round(N * self.sgd_sample)

        sample_rows_ids = random.sample(range(N), batch_size)
        batch_X = X[sample_rows_ids]
        batch_y = y[sample_rows_ids]
        batch_y_pred = batch_X @ self.weights

        grad = (2 / batch_size) * (batch_X.T @ (batch_y_pred - batch_y))
        return grad + self.__get_grad_reg()

    def __calc_loss(self, X: np.array, y: np.array) -> float:
        N = X.shape[0]
        y_pred = X @ self.weights
        loss = (1 / N) * np.linalg.norm(y_pred - y, ord=2) ** 2

        reg_loss = 0
        if self.reg == "l1":          reg_loss = self.l1_coef * np.linalg.norm(self.weights, ord=1)
        if self.reg == "l2":          reg_loss = self.l2_coef * np.linalg.norm(self.weights, ord=2) ** 2
        if self.reg == "elasticnet":  reg_loss = self.l1_coef * np.linalg.norm(self.weights, ord=1) + \
                                                 self.l2_coef * np.linalg.norm(self.weights, ord=2) ** 2

        return loss + reg_loss

    def __calc_metric(self, X: np.array, y: np.array, N: int) -> Optional[float]:
        y_pred = X @ y
        if self.metric == "mae":    return (1 / N) * np.linalg.norm(y_pred - y, ord=1)
        if self.metric == "mse":    return (1 / N) * np.linalg.norm(y_pred - y, ord=2) ** 2
        if self.metric == "rmse":   return math.sqrt((1 / N) * np.linalg.norm(y_pred - y, ord=2) ** 2)
        if self.metric == "mape":   return (1 / N) * np.linalg.norm(((y - y_pred) / y), ord=2) * 100
        if self.metric == "r2":     return 1 - ((np.linalg.norm(y_pred - y, ord=2)) ** 2 / (np.linalg.norm(y - y.mean(), ord=2)) ** 2)
        else:
            return None

    def __write_log(self, iter: int, learning_rate: int | float, mse_loss: float, verbose: int):
        log_str = "{iter} | learning rate: {learning_rate} | mse-loss: {loss_value} | {metric_name}: {metric_value}"
        if iter % verbose == 0:
            print(log_str.format(iter=iter, learning_rate=learning_rate, loss_value=mse_loss, metric_name=self.metric,
                                 metric_value=self.metric_value))

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = 100) -> None:
        random.seed(self.random_state)
        N, D = X.shape

        X, y = X.to_numpy(), y.to_numpy()
        X = np.hstack([np.ones(N).reshape(-1, 1), X])
        self.weights = np.ones(D + 1)

        for iter in range(1, self.n_iter + 1):
            mse_loss = self.__calc_loss(X, y)

            if self.sgd_sample:
                grad = self.__stochastic_gradient_descent(X, y)
            else:
                grad = self.__gradient_descent(X, y)

            learning_rate = self.learning_rate(iter) if callable(self.learning_rate) else self.learning_rate
            self.weights -= (learning_rate * grad)
            self.metric_value = self.__calc_metric(X, y, N) if self.metric else None

            self.__write_log(iter, learning_rate, mse_loss, verbose)

    def get_best_score(self) -> float:
        return self.metric_value

    def get_coef(self) -> np.array:
        return self.weights[1:]

    def predict(self, X: pd.DataFrame) -> np.array:
        N, D = X.shape
        X = np.hstack([np.ones(N).reshape(N - 1, 1), X.to_numpy()])
        return X @ self.weights