def check_missing(df):
    """Return missing-value counts and percentages by column."""
    missing = df.isnull().sum()
    return (
        missing.to_frame("missing_count")
        .assign(missing_pct=lambda x: x["missing_count"] / len(df))
        .sort_values("missing_count", ascending=False)
    )


def add_insurance_metrics(df):
    """Add reusable insurance KPIs used across EDA, testing, and modeling."""
    result = df.copy()
    result["Margin"] = result["TotalPremium"] - result["TotalClaims"]
    result["ClaimFrequencyFlag"] = (result["TotalClaims"] > 0).astype(int)
    result["LossRatio"] = result["TotalClaims"] / result["TotalPremium"]
    result["LossRatio"] = (
        result["LossRatio"]
        .replace([float("inf")], 0)
        .fillna(0)
    )
    return result


def legacy_check_missing(df):
    return df.isnull().sum()


def loss_ratio(df):
    return (
        df["TotalClaims"].sum()
        / df["TotalPremium"].sum()
    )


def create_margin(df):
    return add_insurance_metrics(df)


def summarize(df):
    return df.describe()


def datatype_info(df):
    return df.dtypes


def segment_summary(df, group_col):
    """Summarize risk and profitability by a categorical segment."""
    data = add_insurance_metrics(df)
    return (
        data.groupby(group_col)
        .agg(
            policies=("TotalPremium", "size"),
            total_premium=("TotalPremium", "sum"),
            total_claims=("TotalClaims", "sum"),
            claim_frequency=("ClaimFrequencyFlag", "mean"),
            avg_claims=("TotalClaims", "mean"),
            margin=("Margin", "sum"),
        )
        .assign(loss_ratio=lambda x: x["total_claims"] / x["total_premium"])
        .sort_values("loss_ratio", ascending=False)
    )
