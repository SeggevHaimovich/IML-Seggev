from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float,
                 include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """

        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # the intercept don't suppose to be affected from the lambda so we put
        # 0 instead
        # https://moodle2.cs.huji.ac.il/nu22/mod/forum/discuss.php?d=89494

        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]).T, X]

        if self.lam_ == 0:
            self.coefs_ = np.linalg.pinv(X) @ y
            return

        ##### option 1 #####
        # u, sigma, v = np.linalg.svd(X, full_matrices=False)
        # sigma_lam = sigma / (sigma ** 2 + self.lam_)
        # if self.include_intercept_:
        #     if sigma[0] == 0:
        #         sigma_lam[0] = 0
        #     else:
        #         sigma_lam[0] = 1 / (sigma[0])
        #
        # self.coefs_ = ((v.T * sigma_lam) @ u.T) @ y

        ##### option 2 #####
        lambda_identity = np.identity(X.shape[1])
        if self.include_intercept_:
            lambda_identity[0, 0] = 0
        self.coefs_ = np.linalg.inv(
            X.T @ X + self.lam_ * lambda_identity) @ X.T @ y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            return np.c_[np.ones(X.shape[0]).T, X] @ self.coefs_
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        from ...metrics.loss_functions import mean_square_error
        return mean_square_error(y_true=y, y_pred=self.predict(X))
