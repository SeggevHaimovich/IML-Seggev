from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        squared_error_sum = np.sum((X - self.mu_) ** 2)
        if self.biased_:
            self.var_ = squared_error_sum / len(X)
            # can write:
            # self.var_ = X.var()
        else:
            self.var_ = squared_error_sum / (len(X) - 1)
            # can write:
            # self.var_ = X.var(ddof=1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        return np.exp(-((X - self.mu_) ** 2) / (2 * self.var_)) / np.sqrt(2 * np.pi * self.var_)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        return -len(X) * np.log(2 * np.pi * sigma) / 2 - np.sum((X - mu) ** 2) / (2 * sigma)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = np.mean(X, axis=0)
        # Using the formula from the book
        X_gal = X - self.mu_
        self.cov_ = (X_gal.T @ X_gal) / (X.shape[0] - 1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
                Calculate PDF of observations under Gaussian model with fitted estimators

                Parameters
                ----------
                X: ndarray of shape (n_samples, n_features)
                    Samples to calculate PDF for

                Returns
                -------
                pdfs: ndarray of shape (n_samples, )
                    Calculated values of given samples for PDF function of N(mu_, cov_)

                Raises
                ------
                ValueError: In case function was called prior fitting the model
                """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        X_norm = X - self.mu_
        deter = det(self.cov_)
        inv_cov = inv(self.cov_)
        denominator = np.sqrt(((2 * np.pi) ** len(X[0])) * deter)

        ##### the regular way - but it is slow #####
        # numerator = np.ndarray((X.shape[0],))
        # for i in range(len(numerator)):
        #     line = X_norm[i].reshape((1, len(X_norm[i])))
        #     expo = line @ inv_cov @ X_norm[i]
        #     numerator[i] = np.exp(-0.5 * expo)

        ##### A little better way but pretty slow as well#####
        # def exponent(a):
        #     return np.exp(-0.5 * (a @ inv_cov @ a.T))
        #
        # numerator = np.apply_along_axis(exponent, 1, X_norm)

        ###### A better way I found online #####
        numerator = np.exp(-0.5 * (X_norm * (inv_cov @ X_norm.T).T).sum(-1))
        return numerator / denominator

        # raise NotImplementedError()

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # raise NotImplementedError()
        m = X.shape[0]
        d = X.shape[1]
        deter = det(cov)
        inv_cov = inv(cov)
        part1 = m * (d * np.log(2 * np.pi) + np.log(deter))
        X_norm = X - mu

        ##### the regular way - but it is slow #####
        # part2 = 0
        # for i in range(m):
        #     line = X_norm[i].reshape(1, d)
        #     column = X_norm[i]
        #     part2 += line @ inv_cov @ column

        ##### A little better way but pretty slow as well#####
        # def mat_mult(a):
        #     return a @ inv_cov @ a.T
        # part2 = np.sum(np.apply_along_axis(mat_mult, 1, X_norm))

        ###### A better way I found online #####
        part2 = np.sum((X_norm * (inv_cov @ X_norm.T).T).sum(-1))
        return - (part1 + part2) / 2
