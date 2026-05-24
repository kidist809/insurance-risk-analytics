import pandas as pd


def check_missing(df):
    return df.isnull().sum()


def loss_ratio(df):
    return (
        df["TotalClaims"].sum()
        / df["TotalPremium"].sum()
    )


def create_margin(df):

    df["Margin"] = (
        df["TotalPremium"]
        - df["TotalClaims"]
    )

    return df


def summarize(df):

    return df.describe()


def datatype_info(df):

    return df.dtypes 
