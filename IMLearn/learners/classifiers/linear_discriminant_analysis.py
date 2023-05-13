from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import pandas as pd


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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

        #  todo: without loops
        self.cov_ = np.zeros([X.shape[1], X.shape[1]])

        for i in range(len(df)):
            normalized = (x_df.iloc[i, :] - mu_df.loc[df.iloc[i, -1]]).to_numpy()
            self.cov_ += np.outer(normalized, normalized)

        # X_norm = df.copy()
        # for c in self.classes_:
        #     X_norm.loc[X_norm.y == c, :] -= mu_df.loc[c]
        # X_norm = X_norm.drop('y', axis=1).to_numpy()
        #
        # for i in range(X_norm.shape[0]):
        #     self.cov_ += np.outer(X_norm[i], X_norm[i])

        self._cov_inv = np.linalg.inv(self.cov_)

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
        # for i in range(len(self.classes_)):
        #     a = self._cov_inv @ self.mu_[i].T
        #     b = np.log(self.pi_[i]) - 0.5 * (
        #                 self.mu_[i] @ self._cov_inv @ self.mu_[i].T)
        #     cur = a.T @ X + b
        #     if cur > mx:
        #         mx = cur
        #         index = i
        # return self.classes_[index]

        # better way but more dangerous
        a_T = self.mu_ @ self._cov_inv.T
        b = np.log(self.pi_) - \
            0.5 * ((self.mu_ * (self._cov_inv @ self.mu_.T).T).sum(-1))
        vec = a_T @ X + b
        return self.classes_[np.argmax(vec)]

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
        #  todo I have no idea if it is working, and this is with loop
        #   (i think i have to use loop)
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        like = np.zeros([X.shape[0], len(self.classes_)])
        deter = det(self.cov_)
        for i in range(len(self.classes_)):
            X_norm = X - self.mu_[i]
            denominator = np.sqrt(((2 * np.pi) ** X.shape[1]) * deter)
            numerator = np.exp(
                -0.5 * (X_norm * (self._cov_inv @ X_norm.T).T).sum(-1))
            like[:, i] = (numerator * self.pi_[i]) / denominator
        return like

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
        return misclassification_error(self.predict(X), y)
