import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from IMLearn.metrics.loss_functions import accuracy

IMG_PATH = "..\\images\\Ex4\\adaboost"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    symbols = np.array(["stam", "circle", "x"])

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    def wl(D, S):
        X, y = S[:, :-1], S[:, -1]
        y = y * D
        return DecisionStump().fit(X, y)

    ada = AdaBoost(wl, n_learners)
    ada.fit(train_X, train_y)
    losses_train = []
    losses_test = []
    T = np.arange(1, n_learners + 1)

    for t in T:
        losses_train.append(ada.partial_loss(train_X, train_y, t))
        losses_test.append(ada.partial_loss(test_X, test_y, t))

    fig = go.Figure(data=[go.Scatter(x=T, y=losses_train, name="train"),
                          go.Scatter(x=T, y=losses_test, name="test")],
                    layout=go.Layout(
                        title=f"Train and Test Losses over AdaBoost size with"
                              f" noise {noise}"))
    fig.write_image(os.path.join(IMG_PATH,
                                 f"train_and_test_losses_with_noise_"
                                 f"{noise}.png"),
                    format="png", engine="orca")

    # Question 2: Plotting decision surfaces
    specific_T = [5, 50, 100, 250]

    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=[f"{t} weak learners"
                                        for t in specific_T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(specific_T):
        foo = lambda a: ada.partial_predict(a, t)
        fig.add_traces([decision_surface(foo, lims[0], lims[1],
                                         showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers",
                                   showlegend=False,
                                   marker=dict(color=test_y,
                                               symbol=symbols[
                                                   test_y.astype(int)],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=1, cols=i+1)
    fig.update_layout(
        title=f"Decision Boundary with different numbers of weak learners"
              f" (Decision stumps) with noise {noise}", width=1500, height=500,
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image(os.path.join(IMG_PATH,
                                 f"test_pred_with_different_learners_with"
                                 f"_noise_{noise}.png"),
                    format="png", engine="orca")

    # Question 3: Decision surface of best performing ensemble
    best_num = T[np.argmin(losses_test)]
    foo = lambda a: ada.partial_predict(a, best_num)
    accu = accuracy(test_y, ada.partial_predict(test_X, best_num))
    fig = go.Figure(data=[decision_surface(foo, lims[0], lims[1],
                                           showscale=False),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                     mode="markers",
                                     showlegend=False,
                                     marker=dict(color=test_y,
                                                 symbol=symbols[
                                                     test_y.astype(int)],
                                                 colorscale=[custom[0],
                                                             custom[-1]],
                                                 line=dict(color="black",
                                                           width=1)))],
                    layout=go.Layout(
                        title=f"Decision Boundary with {best_num} "
                              f"weak learners (Decision stump)<br>"
                              f"the best number of weak learners<br> "
                              f"with noise {noise}, "
                              f"accurecy:"
                              f"{accu}",
                        margin=dict(t=100))
                    )
    fig.write_image(
        os.path.join(IMG_PATH, f"best_num_of_learners_with_noise_{noise}.png"),
        format="png", engine="orca")

    # Question 4: Decision surface with weighted samples
    relevant_weights = ada.D_[-1]
    fig = go.Figure(data=[
        decision_surface(ada.predict, lims[0], lims[1],
                         showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                   showlegend=False,
                   marker=dict(color=test_y,
                               symbol=symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1),
                               size=(relevant_weights / np.max(
                                   relevant_weights)) * 30))],
        layout=go.Layout(title=f"Decision Boundary with {T[-1]} "
                               f"weak learners (Decision stump)<br> "
                               f"with noise {noise}, "
                               f"accurecy:"
                               f"{accuracy(test_y, ada.predict(test_X))}",
                         margin=dict(t=100))
    )
    fig.write_image(
        os.path.join(IMG_PATH,
                     f"last_iteration_size_by_weights_with_noise_{noise}.png"),
        format="png", engine="orca")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
