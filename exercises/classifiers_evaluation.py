import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import os

IMG_PATH = "."


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(os.path.join(os.getcwd(), "..\\datasets", f))
        # df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
        # X, results = df.drop('y'), df['y']
        X = data[:, :-1]
        Y = data[:, -1]
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def record_losses(fit: Perceptron, _: np.ndarray, __: int):
            losses.append(fit.loss(X, Y))

        per = Perceptron(callback=record_losses)
        per.fit(X, Y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(
            data=go.Scatter(x=np.arange(len(losses)), y=losses,
                            mode="lines"),
            layout=go.Layout(xaxis=dict(title="number of iterations"),
                             yaxis=dict(title="loss"),
                             title=f"loss as function of number of iterations"
                                   f"<br>{n} case")
        )
        fig.write_image(
            os.path.join(IMG_PATH, f"perceptron_loss-iterations_{n}_case.png"),
            format="png", engine="orca")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        n = f[:-4]
        # Load dataset
        X, y = load_dataset(os.path.join(os.getcwd(), "..\\datasets", f))

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_predict = lda.predict(X)

        naive = GaussianNaiveBayes()
        naive.fit(X, y)
        naive_predict = naive.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accu = round(accuracy(y, lda_predict), 3)
        naive_accu = round(accuracy(y, naive_predict), 3)
        X_df = pd.DataFrame(X, columns=['x1', 'x2'])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Gaussian Naive Bayes<br>accuracy: "
                                f"{naive_accu}",
                                f"Linear Discriminant Analysis<br>accuracy: "
                                f"{lda_accu}"))
        fig.update_layout(title_text=f"<B>DataSet: {n}</b>",
                          margin=dict(t=100), width=1000, height=600,
                          showlegend=False)

        # Add traces for data-points setting symbols and colors
        fig.add_traces([go.Scatter(x=X_df.x1, y=X_df.x2, mode='markers',
                                   marker=dict(color=naive_predict, symbol=y)),
                        go.Scatter(x=X_df.x1, y=X_df.x2, mode='markers',
                                   marker=dict(color=lda_predict, symbol=y))
                        ], rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces(
            [go.Scatter(x=naive.mu_[:, 0], y=naive.mu_[:, 1], mode='markers',
                        marker=dict(color='black', symbol='x', size=10)),
             go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode='markers',
                        marker=dict(color='black', symbol='x', size=10))
             ], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.mu_)):
            fig.add_traces([get_ellipse(naive.mu_[i], np.diag(naive.vars_[i])),
                            get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])
        fig.write_image(os.path.join(IMG_PATH, f"LDA_Naive_{n}.png"),
                        format="png", engine="orca")


if __name__ == '__main__':
    np.random.seed(0)
    X, y = load_dataset("..\\datasets\\gaussian1.npy")
    run_perceptron()
    compare_gaussian_classifiers()
