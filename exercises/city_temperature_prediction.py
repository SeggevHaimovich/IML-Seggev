import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"

DAYS_PER_MONTH = [31, 31 + 30, 31 + 30 + 31, 31 + 30 + 31 + 30]


def _deal_with_invalid_data(df: pd.DataFrame):
    mask = df[
        (df["Day"] <= 0) | (df["Day"] > 31) | (df["Month"] <= 0) | (df["Month"] > 12) |
        ((df["Day"] > 29) & (df["Month"] == 2)) |
        ((df["Day"] > 30) & ((df["Month"] == 4) | (df["Month"] == 6) | (df["Month"] == 9) | (df["Month"] == 11))) |
        (df["Temp"] < -50)].index
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
    df = _deal_with_invalid_data(df)

    ones = pd.Series(np.ones(df.shape[0], dtype=str))
    years = df.Year.apply(str)
    first_dates = pd.Series(years + "-" + ones + "-" + ones)
    first_dates = pd.to_datetime(first_dates, format="%Y-%m-%d")
    df["DayOfYear"] = (df.Date - first_dates).dt.days + 1
    # df["DayOfYear"] = df.apply(lambda a: (a.Date - pd.to_datetime(f"{a.Year}-01-01", format="%Y-%m-%d")).days + 1, axis=1)
    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset

    df = load_data("C:/Users/segge/source/repos/IML-Seggev/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country

    il_data = df[df["Country"] == "Israel"].reset_index(drop=True)
    il_data["StrYear"] = il_data["Year"].astype(str)
    fig = px.scatter(il_data, x="DayOfYear", y="Temp", color="StrYear", title="blabla")
    fig.write_image("C:/Users/segge/source/repos/IML-Seggev/images/Ex2/trying.png", format="png", engine="orca")

    per_month = il_data.groupby("Month").Temp.agg(['var', 'mean', 'std'])
    fig = px.bar(per_month, y="std")
    fig.write_image("C:/Users/segge/source/repos/IML-Seggev/images/Ex2/bar.png", format="png", engine="orca")


    # Question 3 - Exploring differences between countries

    groups = df.groupby(["Country", "Month"]).Temp.agg(['std', 'mean']).reset_index(drop=True)
    fig = px.line(groups, x="Month", y='mean', color="Country", error_y="std")
    fig.write_image("C:/Users/segge/source/repos/IML-Seggev/images/Ex2/Q3.png", format="png", engine="orca")


    # Question 4 - Fitting model for different values of `k`

    proportion = 0.75
    train_X, train_y, test_X, test_y = split_train_test(il_data.DayOfYear, il_data.Temp, train_proportion=proportion)

    degrees = np.arange(1, 11)
    losses = []
    for k in degrees:
        pm = PolynomialFitting(k)
        pm.fit(train_X.to_numpy(), train_y.to_numpy())
        losses.append(pm.loss(test_X.to_numpy(), test_y.to_numpy()))
    for i, loss in enumerate(losses):
        print("the loss for k = ", i, " is ", loss)
    fig = go.Figure(data=go.Bar(x=degrees, y=losses),
                    layout=go.Layout(title="Loss as function of degree",
                                     xaxis_title="degree", yaxis_title="Loss"))
    fig.write_image("C:/Users/segge/source/repos/IML-Seggev/images/Ex2/Q4.png", format="png", engine="orca")

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
    not_il_data = df.drop(df[df.Country == "Israel"].index).reset_index(drop=True)

    chosen_k = 4
    losses = []
    pm = PolynomialFitting(chosen_k)
    pm.fit(il_data.DayOfYear.to_numpy(), il_data.Temp.to_numpy())

    countries = not_il_data.Country.unique()
    for country in countries:
        country_data = not_il_data[not_il_data.Country == country]
        losses.append(round(pm.loss(country_data.DayOfYear, country_data.Temp), 2))
    fig = go.Figure(data=go.Bar(x=countries, y=losses, text=losses),
                    layout=go.Layout(
                        title=f"Loss as function of "
                              f"country with k = {chosen_k}",
                        xaxis_title="Country", yaxis_title="Loss"))
    fig.write_image("C:/Users/segge/source/repos/IML-Seggev/images/Ex2/Q5.png", format="png", engine="orca")
    print("finish :)")
