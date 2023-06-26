import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from plotly.subplots import make_subplots

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics import mean_square_error
from IMLearn.metrics import misclassification_error
import plotly.graph_objects as go
import os

IMAGE_PATH = "..\\images\\Ex5"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_list, weights_list = [], []

    def callback(solver, weights, val, grad, t, eta, delta):
        values_list.append(val)
        weights_list.append(weights)

    return callback, values_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    min_val_l1, min_val_l2 = np.inf, np.inf
    for eta in etas:
        lr = FixedLR(eta)
        callback1, values1, weights1 = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback1)
        l1 = L1(init)
        gd.fit(l1, None, None)
        cur_min_l1 = np.min(values1)
        if cur_min_l1 < min_val_l1:
            min_val_l1 = cur_min_l1

        callback2, values2, weights2 = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback2)
        l2 = L2(init)
        gd.fit(l2, None, None)
        cur_min_l2 = np.min(values2)
        if cur_min_l2 < min_val_l2:
            min_val_l2 = cur_min_l2

        fig = plot_descent_path(L1, np.array(weights1), title=f"Path of GD of L1 and size of step: {eta}")
        fig.write_image(os.path.join(os.getcwd(), IMAGE_PATH, f"L1_fixed_weights_{eta}.png"), format="png",
                        engine='orca')
        fig = plot_descent_path(L2, np.array(weights2), title=f"Path of GD of L2 and size of step: {eta}")
        fig.write_image(os.path.join(os.getcwd(), IMAGE_PATH, f"L2_fixed_weights_{eta}.png"), format="png",
                        engine='orca')

        fig = go.Figure(go.Scatter(x=np.arange(len(values1)), y=values1),
                        go.Layout(title=f"Norm L1 as a function of gd iteration step: {eta}", xaxis_title="iteration",
                                  yaxis_title="norm"))
        fig.write_image(os.path.join(os.getcwd(), IMAGE_PATH, f"L1_fixed_values_{eta}.png"), format="png",
                        engine='orca')
        fig = go.Figure(go.Scatter(x=np.arange(len(values2)), y=values2),
                        go.Layout(title=f"Norm L2 as a function of gd iteration step: {eta}", xaxis_title="iteration",
                                  yaxis_title="norm"))
        fig.write_image(os.path.join(os.getcwd(), IMAGE_PATH, f"L2_fixed_values_{eta}.png"), format="png",
                        engine='orca')
    print("Lowest loss for L1 is: ", min_val_l1)
    print("Lowest loss for L2 is: ", min_val_l2)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    callback, weights, values = get_gd_state_recorder_callback()
    gd = GradientDescent(callback=callback)
    log_reg = LogisticRegression(solver=gd)
    log_reg.fit(X_train.to_numpy(), y_train.to_numpy())
    y_prob = log_reg.predict_proba(X_train.to_numpy())

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    from utils import custom
    c = [custom[0], custom[-1]]
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.write_image(os.path.join(os.getcwd(), IMAGE_PATH, f"roc_curve.png"), format="png",
                    engine='orca')

    best_index = np.argmax(tpr - fpr)
    best_alpha = fpr[best_index]

    log_reg.alpha_ = best_alpha
    print("Question 9: The best alpha is: ", best_alpha,
          " and the test error using this alpha is: ", log_reg.loss(X_test.to_numpy(), y_test.to_numpy()))



    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    gd = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000, callback=callback)
    lam_vals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    valid_losses_l1, train_losses_l1, valid_losses_l2, train_losses_l2 = [], [], [], []
    for lam in lam_vals:
        log_reg = LogisticRegression(solver=gd, penalty='l1', lam=lam, alpha=0.5)
        train, valid = cross_validate(log_reg, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)
        train_losses_l1.append(train)
        valid_losses_l1.append(valid)

        log_reg = LogisticRegression(solver=gd, penalty='l2', lam=lam, alpha=0.5)
        train, valid = cross_validate(log_reg, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)
        train_losses_l2.append(train)
        valid_losses_l2.append(valid)

    best_lam_l1 = lam_vals[np.argmin(valid_losses_l1)]
    log_reg_l1_best = LogisticRegression(solver=gd, penalty='l1', lam=best_lam_l1, alpha=0.5)
    log_reg_l1_best.fit(X_train.to_numpy(), y_train.to_numpy())
    test_error_l1 = log_reg_l1_best.loss(X_test.to_numpy(), y_test.to_numpy())
    print("Question 10: The best lambda to l1 penalty is: ", best_lam_l1, " with test error: ", test_error_l1)

    best_lam_l2 = lam_vals[np.argmin(valid_losses_l2)]
    log_reg_l2_best = LogisticRegression(solver=gd, penalty='l2', lam=best_lam_l2, alpha=0.5)
    log_reg_l2_best.fit(X_train.to_numpy(), y_train.to_numpy())
    test_error_l2 = log_reg_l2_best.loss(X_test.to_numpy(), y_test.to_numpy())
    print("Question 11: The best lambda to l2 penalty is: ", best_lam_l2, " with test error: ", test_error_l2)


    # fig = make_subplots(rows=1, cols=2, subplot_titles=["l1", "l2"])
    #
    # fig.add_traces([go.Scatter(x=lam_vals, y=train_losses_l1, mode="lines", name="l1 train"),
    #                 go.Scatter(x=lam_vals, y=valid_losses_l1, mode="lines", name="l1 valid"),
    #                 go.Scatter(x=lam_vals, y=train_losses_l2, mode="lines", name="l2 train"),
    #                 go.Scatter(x=lam_vals, y=valid_losses_l2, mode="lines", name="l2 valid"),
    #                 ], rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])
    # fig.update_layout(
    #     title="Different Regularization parameter - l1 & l2",
    #     width=800, height=500, margin=dict(t=100))
    # fig.update_xaxes(title_text=r"$\lambda \text{ value}$", row=1, col=1)
    # fig.update_xaxes(title_text=r"$\lambda \text{ value}$", row=1, col=2)
    # fig.update_yaxes(title_text="Loss", row=1, col=1)
    # fig.write_image(os.path.join(IMAGE_PATH, "l1_&_l2_lambda.png"),
    #                 format="png", engine="orca")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
