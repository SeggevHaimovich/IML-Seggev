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
        self.vars_ = X_norm.groupby(X_norm.y).mean().to_numpy()
        bla = 5

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
        return self.classes_[np.argmax(self.likelihood(X), 1)]

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
        temp1 = np.ones_like(self.vars_)
        temp1[self.vars_ == 0] = 0
        temp2 = self.vars_.copy()
        temp2[temp2 == 0] = 1
        vars_inv = temp1 / temp2

        like = np.zeros([X.shape[0], len(self.classes_)])
        deter = self.vars_.prod(1)
        for i in range(len(self.classes_)):
            X_norm = X - self.mu_[i]
            denominator = np.sqrt(((2 * np.pi) ** X.shape[1]) * deter[i])
            numerator = np.exp(
                -0.5 * (X_norm * (np.diag(vars_inv[i]) @ X_norm.T).T).sum(-1))
            # (X_norm * (self._cov_inv @ X_norm.T).T).sum(-1))
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
        return misclassification_error(y, self.predict(X))
