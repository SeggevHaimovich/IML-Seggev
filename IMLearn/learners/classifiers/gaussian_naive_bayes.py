from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import pandas as pd


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / X.shape[0]
        x_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y, columns=["y"])
        df = pd.concat([x_df, y_df], axis=1)
        mu_df = df.groupby(df.y).mean()
        self.mu_ = mu_df.to_numpy()

        X_norm = df.copy()
        for c in self.classes_:
            X_norm.loc[X_norm.y == c, :] -= mu_df.loc[c]
        X_norm_no_y = X_norm.drop('y', axis=1)
        X_norm_no_y = X_norm_no_y ** 2
        X_norm = pd.concat([X_norm_no_y, y_df], axis=1)
        vars = X_norm.groupby(X_norm.y).mean().to_numpy()
        nps = df.groupby(df.y)['y'].count().to_numpy()
        self.vars_ = vars / (nps.reshape([nps.shape[0], -1]))

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
        # mx = -np.inf
        # index = -1
        # psudo_inv = 1
        # for i in range(len(self.classes_)):
        #     a = self._cov_inv @ self.mu_[i].T
        #     b = np.log(self.pi_[i]) - 0.5 * (
        #                 self.mu_[i] @ self._cov_inv @ self.mu_[i].T)
        #     cur = a.T @ X + b
        #     if cur > mx:
        #         mx = cur
        #         index = i
        # return self.classes_[index]
    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        raise NotImplementedError()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        raise NotImplementedError()
