import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

DAYS_PER_MONTH = [31,31+30,31+30+31,31+30+31+30]


def deal_with_invalid_data(df: pd.DataFrame):
    mask = df[(df["Day"] <= 0) | (df["Month"] <= 0) | (df["Month"] > 12) |
              ((df["Day"] > 29) & (df["Month"] == 2)) |
              ((df["Day"] > 30) & (df["Month"] in [4, 6, 9, 11]))].index
    return df.drop(mask, axis=0).reset_index(drop=True)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # raise NotImplementedError()
    df = pd.read_csv(filename, parse_dates=[2])
    ones = pd.Series(np.ones(df.shape[0]))
    ones = ones.apply(int).apply(str)
    years = df.Year.apply(str)
    first_dates = pd.Series(years + ones + ones)
    first_dates = pd.to_datetime(first_dates, format="%Y%m%d")
    # deal_with_invalid_data(df)
    df["DayOfYear"] = (df.Date - first_dates).dt.days + 1
    # df["DayOfYear"] = df.apply(lambda a: (a.Date - pd.to_datetime(f"{a.Year}-01-01", format="%Y-%m-%d")).days + 1, axis=1)
    print("hii")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    # raise NotImplementedError()
    load_data("C:/Users/segge/source/repos/IML-Seggev/datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
