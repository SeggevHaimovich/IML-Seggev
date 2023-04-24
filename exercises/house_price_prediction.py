from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

ZIPCODE = "zipcode"
PRICE = "price"
SQFT_ABOVE_PERCENTAGE = "sqft_above_percentage"
LONG = "long"
LAT = "lat"
ID = "id"
YR_RENOVATED = "yr_renovated"
YR_BUILT = "yr_built"
DATE = "date"
SQFT_BASEMENT = "sqft_basement"
SQFT_LIVING15 = "sqft_living15"

CSV_PATH = "C:/Users/segge/source/repos/IML-Seggev/datasets/house_prices.csv"
IMG_PATH = "C:/Users/segge/source/repos/IML-Seggev/images/Ex2/houses/"

after_preprocessing_columns = None
means = None
preprocess_train = False


def change_columns(X: pd.DataFrame):
    def date_change_func(a):
        if type(a) == str and len(a) >= 8:
            return int(a[:8])
        return np.nan

    processed = X.drop([ID, LAT, LONG], axis=1)

    processed.loc[:, 'date'] = processed.date.apply(date_change_func)

    # processed[YR_RENOVATED] = X[[YR_BUILT, YR_RENOVATED]].max(axis=1)

    # todo understand why not working
    # mask_date = processed[(processed.date == '0') | (processed.date.isnull())].index
    # processed.loc[mask_date, 'date'] = 0
    # processed.loc[processed.drop(mask_date).index, 'date'] = processed.date.apply((lambda a: int(a[:8])))

    processed[SQFT_ABOVE_PERCENTAGE] = pd.Series(np.zeros(processed.shape[0]))
    processed.loc[(processed.sqft_living > 0), SQFT_ABOVE_PERCENTAGE] = processed.sqft_above / processed.sqft_living

    processed = pd.get_dummies(processed, prefix_sep='=', columns=["zipcode"])
    processed = processed.reindex(columns=after_preprocessing_columns, fill_value=0)
    # processed = processed.fillna(0)
    return processed.reset_index(drop=True)


def make_wrong_vals_nan(X: pd.DataFrame):
    X.replace('nan', np.nan, inplace=True)
    X[X < 0] = np.nan
    X.loc[X.bedrooms == 0, 'bedrooms'] = np.nan
    X.loc[X.bathrooms == 0, 'bathrooms'] = np.nan
    X.loc[(X.sqft_living == 0) | (X.sqft_above > X.sqft_living) |
          (X.sqft_lot < X.sqft_living), 'sqft_living'] = np.nan
    X.loc[(X.sqft_lot == 0) | (X.sqft_lot < X.sqft_living), 'sqft_lot'] = \
        np.nan
    X.loc[X.floors == 0, 'floors'] = np.nan
    X.loc[(X.waterfront != 0) & (X.waterfront != 1), 'waterfront'] = np.nan
    X.loc[X.view > 4, 'view'] = np.nan
    X.loc[(X.condition < 1) | (X.condition > 5), 'condition'] = np.nan
    X.loc[(X.grade < 1) | (X.grade > 13), 'grade'] = np.nan
    X.loc[X.sqft_above > X.sqft_living, 'sqft_above'] = np.nan
    X.loc[X.yr_renovated > X.yr_built, ['yr_built', 'yr_renovated']] = np.nan
    X.loc[X.sqft_living15 > X.sqft_lot15, ['sqft_living15', 'sqft_lot15']] = np.nan
    return X


def change_rows_train(X: pd.DataFrame, y: pd.Series):
    mask = X[X.isnull().any(1)].index
    X = X.drop(mask).reset_index(drop=True)
    y = y.drop(mask).reset_index(drop=True)
    return X, y


def change_rows_test(X: pd.DataFrame):
    for column in X:
        X.loc[X[column].isnull(), column] = means[column]
    return X


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
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    global after_preprocessing_columns, means, preprocess_train
    X = change_columns(X)
    X = make_wrong_vals_nan(X)
    if y is not None:
        after_preprocessing_columns = X.columns
        preprocess_train = True
        means = pd.Series(data=np.mean(X, axis=0), index=X.columns)
        X, y = change_rows_train(X, y)
        return X.reset_index(drop=True), y.reset_index(drop=True)
    else:
        if not preprocess_train:
            print("you have to preprocess the train set before preprocessing the test set")
            exit(1)
        X = change_rows_test(X)
        return X.reset_index(drop=True)

    # raise NotImplementedError()


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
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
        title = f"Relation between {column_vec.name} and {y.name}<br>the Pearson Correlation is {pearson}"
        fig = px.scatter(x=column_vec, y=y, title=title, labels={"x": column_name, "y": "price"})
        fig.write_image(IMG_PATH + column_vec.name + ".png", format="png", engine='orca')
        print(column_name)


if __name__ == '__main__':
    # todo: ratio between neighbors and self?
    # todo cancel rows with too big values
    np.random.seed(0)
    df = pd.read_csv(CSV_PATH)
    mask = df[(df.price <= 0) | (df.price.isnull())].index
    df = df.drop(mask, axis=0).reset_index(drop=True)
    df.replace('nan', np.nan, inplace=True)

    ########################### play ################################
    # change_columns(df)
    # make_wrong_vals_nan(df.drop('date', axis=1))
    # df.dropna(inplace=True)
    # df.date = df.date.apply(date_change_func)
    # df.drop([ID, LAT, LONG], axis=1, inplace=True)
    # # df = pd.get_dummies(df, prefix_sep='=', columns=[ZIPCODE])
    # df["yr_renovated_new"] = df[[YR_BUILT, YR_RENOVATED]].max(axis=1)
    # df[SQFT_ABOVE_PERCENTAGE] = pd.Series(np.zeros(df.shape[0]))
    # df.loc[(df.sqft_living > 0), SQFT_ABOVE_PERCENTAGE] = df.sqft_above / df.sqft_living
    # var_price = df.price.var()
    # for column_name in df.drop('price', axis=1):
    #     column_vec = df[column_name]
    #     var_column = column_vec.var()
    #     if not var_column:
    #         print(f"column {column_name} has no var")
    #     pearson = column_vec.cov(df.price) / np.sqrt(var_column * var_price)
    #     print(f"{column_name} - {pearson}")

    # checks:
    # Question 1 - split data into train and test sets
    proportion = 0.75
    train_X, train_y, test_X, test_y = split_train_test(df.drop(df[[PRICE]], axis=1), df[PRICE], train_proportion=proportion)

    # # Question 2 - Preprocessing of housing prices dataset
    processed_train_X, processed_train_y = preprocess_data(train_X, train_y)
    processed_train = pd.concat([processed_train_X, processed_train_y], axis=1)
    processed_test = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response

    feature_evaluation(processed_train_X, processed_train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression()
    var_pred, mean_pred = [], []
    percentages = np.arange(10, 101)
    var_y = test_y.var()
    for p in percentages:
        predictions_per_p = []
        for _ in range(10):
            picked = processed_train.sample(frac=p / float(100), replace=False)
            lr._fit(picked.drop(PRICE, axis=1).to_numpy(), picked.price.to_numpy())
            predictions_per_p.append(lr._loss(processed_test.to_numpy(), test_y.to_numpy()))
        var_pred.append(np.sqrt(np.var(predictions_per_p)))
        mean_pred.append(np.mean(predictions_per_p))
        print(p)
        print(mean_pred[-1] / var_y)
    mean_pred, var_pred = np.array(mean_pred), np.array(var_pred)
    fig = go.Figure(data=[go.Scatter(x=percentages, y=mean_pred, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                          go.Scatter(x=percentages, y=mean_pred - 2 * var_pred, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=percentages, y=mean_pred + 2 * var_pred, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(title="Loss as function of training set's size"))
    fig.write_image(IMG_PATH + "last.png", format="png", engine='orca')
    print("finish :)")
