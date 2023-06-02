from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

IMG_PATH = "..\\images\\Ex4\\model_selection"


def load_data(n_train):
    X, y = datasets.load_diabetes(return_X_y=True)
    indices = np.random.choice(len(y), n_train)
    return X[indices], y[indices], np.delete(X, indices, axis=0), \
        np.delete(y, indices)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    train_X, train_y, test_X, test_y = load_data(n_samples)

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    start_val, end_val = 0.001, 2
    lam_vals = np.linspace(start_val, end_val, num=n_evaluations)
    ridge_valid_losses, ridge_train_losses, lasso_valid_losses, \
        lasso_train_losses = [], [], [], []
    for lam in lam_vals:
        train, valid = cross_validate(RidgeRegression(lam), train_X, train_y,
                                      mean_square_error, 5)
        ridge_train_losses.append(train)
        ridge_valid_losses.append(valid)
        train, valid = cross_validate(Lasso(alpha=lam), train_X, train_y,
                                      mean_square_error, 5)
        lasso_train_losses.append(train)
        lasso_valid_losses.append(valid)

    fig = make_subplots(rows=1, cols=2,
                        horizontal_spacing=0.01, vertical_spacing=.03)

    fig.add_traces([go.Scatter(x=lam_vals, y=ridge_train_losses, mode="lines",
                               name="ridge train"),
                    go.Scatter(x=lam_vals, y=ridge_valid_losses, mode="lines",
                               name="ridge valid"),
                    go.Scatter(x=lam_vals, y=lasso_train_losses, mode="lines",
                               name="lasso train"),
                    go.Scatter(x=lam_vals, y=lasso_valid_losses, mode="lines",
                               name="lasso valid"),
                    ],
                   rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])
    fig.update_layout(
        title="bla", width=1500, height=500,
        margin=dict(t=100))
    fig.write_image(os.path.join(IMG_PATH, "bla.png"),
                    format="png", engine="orca")

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lam_vals[np.argmin(ridge_valid_losses)]
    best_lam_lasso = lam_vals[np.argmin(lasso_valid_losses)]
    ridge_error = RidgeRegression(best_lam_ridge).fit(test_X, test_y).loss(
        test_X, test_y)
    lasso_model = Lasso(alpha=best_lam_ridge).fit(test_X, test_y)
    lasso_error = mean_square_error(y_true=test_y,
                                    y_pred=test_X @ lasso_model.coef_)
    lr_error = LinearRegression().fit(test_X, test_y).loss(test_X, test_y)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter(n_samples=50, n_evaluations=5000)
    print("finish")
