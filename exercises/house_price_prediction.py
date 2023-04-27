from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os

pio.templates.default = "simple_white"

ZIP = "zipcode"
PRICE = "price"
ABOVE_PER = "sqft_above_percentage"
LONG = "long"
LAT = "lat"
ID = "id"
RENOVATED = "yr_renovated"
BUILT = "yr_built"
DATE = "date"
BASEMENT = "sqft_basement"
LIVING15 = "sqft_living15"
LIVING_PER = "sqft_living_percentage_15"
LOT_PER = "sqft_lot_percentage_15"
FLOORS = 'floors'
WATERFRONT = 'waterfront'
COND = 'condition'
GRADE = 'grade'
LIVING = 'sqft_living'
LOT = 'sqft_lot'
ABOVE = 'sqft_above'
BEDROOMS = 'bedrooms'
BATH = 'bathrooms'
LOT15 = 'sqft_lot15'

DATASET = os.path.join(os.getcwd(), "..\\datasets\\house_prices.csv")
# IMG_PATH = os.path.join(os.getcwd(), "..\\images\\Ex2\\Houses")

after_preprocessing_columns = None
means = None
preprocess_train = False
ZERO_DATE = pd.to_datetime("2014-01-01", format="%Y-%m-%d")


def change_columns(X: pd.DataFrame):
    """
    changing the columns of the training and the test sets to be better for
    fitting our model

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    Returns
    -------
    Dataframe with relevant columns for studying the price of the houses
    """

    processed = X.drop([ID, LAT, LONG], axis=1)

    # processed[YR_RENOVATED] = X[[YR_BUILT, YR_RENOVATED]].max(axis=1)

    processed[ABOVE_PER] = pd.Series(np.zeros(processed.shape[0]))
    processed.loc[(processed.sqft_living > 0), ABOVE_PER] = \
        processed.sqft_above / processed.sqft_living

    processed[LIVING_PER] = pd.Series(np.zeros(processed.shape[0]))
    processed.loc[(processed.sqft_living15 > 0), LIVING_PER] = \
        processed.sqft_living / processed.sqft_living15

    processed[LOT_PER] = pd.Series(np.zeros(processed.shape[0]))
    processed.loc[(processed.sqft_lot15 > 0), LOT_PER] = \
        processed.sqft_lot / processed.sqft_lot15

    processed = pd.get_dummies(processed, prefix_sep='=', columns=[ZIP])
    return processed


def make_wrong_vals_nan(X: pd.DataFrame):
    """
    Change all the spots with wrong values to be null values, so we can change
    them later easily

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    Returns
    -------
    Dataframe with null values instead of wrong ones
    """
    X.replace('nan', np.nan, inplace=True)
    X[X < 0] = np.nan
    X.loc[X.floors == 0, FLOORS] = np.nan
    X.loc[~X.waterfront.isin([0, 1]), WATERFRONT] = np.nan
    X.loc[~X.view.isin(range(0, 5)), 'view'] = np.nan
    X.loc[~X.condition.isin(range(1, 6)) | (X.condition > 5), COND] = np.nan
    X.loc[~X.grade.isin(range(1, 14)), GRADE] = np.nan
    X.loc[(X.sqft_living == 0) |
          (X.sqft_above > X.sqft_living), LIVING] = np.nan
    X.loc[(X.sqft_lot == 0), LOT] = np.nan
    X.loc[X.sqft_above > X.sqft_living, ABOVE] = np.nan
    X.loc[(X.yr_renovated != 0) & (X.yr_renovated < X.yr_built), [BUILT,
                                                                  RENOVATED]] = np.nan
    X.loc[X.bedrooms == 0, BEDROOMS] = np.nan
    X.loc[X.bathrooms == 0, BATH] = np.nan
    X.loc[X.sqft_living15 == 0, LIVING15] = np.nan
    X.loc[X.sqft_lot15 == 0, LOT15] = np.nan

    return X


def change_rows_train(X: pd.DataFrame, y: pd.Series):
    """
    delete all the lines that contains null values in the training matrix

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Dataframe and it's matching price vector without null values
    """
    mask = X[X.isnull().any(1)].index
    X = X.drop(mask).reset_index(drop=True)
    y = y.drop(mask).reset_index(drop=True)
    return X, y


def change_rows_test(X: pd.DataFrame):
    """
    Converts the null values in the test matrix to the mean values of the
    training set

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem


    Returns
    -------
    Dataframe without null values
    """
    for column in X:
        X.loc[X[column].isnull(), column] = means[column]
    return X


def make_vals_numeric(X: pd.DataFrame):
    """
    changing the type of the values in the dataframe to numeric values
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    Returns
    -------
    numeric Dataframe
    """

    def date_change_func(a):
        if type(a) == str and len(a) >= 8:
            return a[:8]
        return np.nan

    X.loc[:, DATE] = X.date.apply(date_change_func)
    X.loc[:, DATE] = X.date.apply(
        lambda a: (pd.to_datetime(a, format="%Y%m%d") - ZERO_DATE).days
        if type(a) == str else a)

    return X.apply(pd.to_numeric, errors='coerce')


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a
    single
    DataFrame or a Tuple[DataFrame, Series]
    """
    global after_preprocessing_columns, means, preprocess_train
    X = make_vals_numeric(X)
    X = change_columns(X)
    X = make_wrong_vals_nan(X)
    if y is not None:
        X, y = change_rows_train(X, y)
        after_preprocessing_columns = X.columns
        preprocess_train = True
        means = pd.Series(data=np.mean(X, axis=0), index=X.columns)
        return X.reset_index(drop=True), y.reset_index(drop=True)
    else:
        if not preprocess_train:
            print("you have to preprocess the train set before preprocessing "
                  "the test set")
            exit(1)
        X = X.reindex(columns=after_preprocessing_columns, fill_value=0)
        X = change_rows_test(X)
        return X.reset_index(drop=True)


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    var_y = y.var()
    for column_name in X:
        column_vec = X[column_name]
        var_column = column_vec.var()
        if not var_column:
            continue
        pearson = column_vec.cov(y) / np.sqrt(var_column * var_y)
        title = f"Relation between {column_vec.name} and " \
                f"{y.name}<br>the Pearson Correlation is {pearson}"
        fig = px.scatter(x=column_vec, y=y, title=title,
                         labels={"x": column_name, "y": PRICE})
        fig.write_image(os.path.join(output_path, column_vec.name + ".png"),
                        format="png", engine='orca')
        # print(column_name)  # to show the process working


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv(DATASET)
    df.replace('nan', np.nan, inplace=True)

    # deleting invalid price rows
    mask = df[(df.price <= 0) | (df.price.isnull())].index
    df = df.drop(mask, axis=0).reset_index(drop=True)

    # Question 1 - split data into train and test sets
    proportion = 0.75
    train_X, train_y, test_X, test_y = \
        split_train_test(df.drop(df[[PRICE]], axis=1),
                         df[PRICE], train_proportion=proportion)

    # # Question 2 - Preprocessing of housing prices dataset
    processed_train_X, processed_train_y = preprocess_data(train_X, train_y)
    processed_train = pd.concat([processed_train_X, processed_train_y], axis=1)
    processed_test = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response

    feature_evaluation(processed_train_X, processed_train_y)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)
    lr = LinearRegression()
    var_pred, mean_pred = [], []
    percentages = np.arange(10, 101)
    var_y = test_y.var()
    for p in percentages:
        predictions_per_p = []
        for _ in range(10):
            picked = processed_train.sample(frac=p / float(100), replace=False)
            lr.fit(picked.drop(PRICE, axis=1).to_numpy(),
                   picked.price.to_numpy())
            predictions_per_p.append(lr.loss(processed_test.to_numpy(),
                                             test_y.to_numpy()))
        var_pred.append(np.sqrt(np.var(predictions_per_p)))
        mean_pred.append(np.mean(predictions_per_p))
        # print(p)
        # print(mean_pred[-1] / var_y)  # used to check if my loss is good
    mean_pred, var_pred = np.array(mean_pred), np.array(var_pred)
    fig = go.Figure(
        data=[go.Scatter(x=percentages, y=mean_pred, mode="markers+lines",
                         name="Mean Prediction", line=dict(dash="dash"),
                         marker=dict(color="green", opacity=.7)),
              go.Scatter(x=percentages, y=mean_pred - 2 * var_pred, fill=None,
                         mode="lines", line=dict(color="lightgrey"),
                         showlegend=False),
              go.Scatter(x=percentages, y=mean_pred + 2 * var_pred,
                         fill='tonexty', mode="lines",
                         line=dict(color="lightgrey"), showlegend=False)],
        layout=go.Layout(title="Loss as function of training set's size",
                         xaxis_title="% of Train set", yaxis_title="MSE"))
    fig.write_image("Losses_graph.png", format="png", engine='orca')
