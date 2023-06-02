from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    ### random choice ###
    # rand_indices = np.random.permutation(np.arange(len(y)))
    # parts = np.array_split(rand_indices, cv)

    ### non-random choice ###
    parts = np.array_split(np.arange(len(y)), cv)

    train_loss = 0
    valid_loss = 0
    for i in range(cv):
        train_X = np.delete(X, parts[i], axis=0)
        train_y = np.delete(y, parts[i], axis=0)
        test_X = X[parts[i]]
        test_y = y[parts[i]]
        cur_estimator = estimator.fit(train_X, train_y)
        train_loss += scoring(cur_estimator.predict(train_X), train_y)
        valid_loss += scoring(cur_estimator.predict(test_X), test_y)
    return train_loss / cv, valid_loss / cv
