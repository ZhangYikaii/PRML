import numpy as np
from prml.linear.logistic_regression import LogisticRegression


class BayesianLogisticRegression(LogisticRegression):
    """
    Logistic regression model

    w ~ Gaussian(0, alpha^(-1)I)
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def __init__(self, alpha:float=1.):
        self.alpha = alpha

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """
        bayesian estimation of logistic regression model
        using Laplace approximation

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
            binary 0 or 1
        max_iter : int, optional
            maximum number of paramter update iteration (the default is 100)
        """
        ### 请注意: 这里好像是在用迭代重加权计算w.
        w = np.zeros(np.size(X, 1))
        eye = np.eye(np.size(X, 1))
        self.w_mean = np.copy(w)

        # 先验中的精度 S_0^{-1}:
        self.w_precision = self.alpha * eye
        for _ in range(max_iter):
            w_prev = np.copy(w)
            # 式(4.143)中 y_n = \sigma(w^T \phi_n)
            y = self._sigmoid(X @ w)
            #
            grad = X.T @ (y - t) + self.w_precision @ (w - self.w_mean)
            # 式(4.143):
            hessian = (X.T * y * (1 - y)) @ X + self.w_precision
            try:
                # 似乎是迭代重加权最小平方
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian

    def proba(self, X:np.ndarray):
        """
        compute probability of input belonging class 1

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable

        Returns
        -------
        (N,) np.ndarray
            probability of positive
        """
        mu_a = X @ self.w_mean
        var_a = np.sum(np.linalg.solve(self.w_precision, X.T).T * X, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
