import numpy as np
import pandas as pd
from scipy import stats


def add_testing_metrics(df):
    """Create metrics used in the A/B hypothesis tests."""
    result = df.copy()
    result["ClaimFlag"] = (result["TotalClaims"] > 0).astype(int)
    result["Margin"] = result["TotalPremium"] - result["TotalClaims"]
    return result


def chi_square_claim_frequency(df, segment_col):
    """Test whether claim frequency differs across categories."""
    data = add_testing_metrics(df)
    table = pd.crosstab(data[segment_col], data["ClaimFlag"])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    return {
        "test": "chi-square",
        "segment": segment_col,
        "statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "contingency_table": table,
        "expected": expected,
    }


def two_proportion_z_test(df, segment_col, control, test):
    """Compare claim frequency for two selected categories."""
    data = add_testing_metrics(df)
    subset = data[data[segment_col].isin([control, test])]
    grouped = subset.groupby(segment_col)["ClaimFlag"].agg(["sum", "count"])

    claims_a = grouped.loc[control, "sum"]
    n_a = grouped.loc[control, "count"]
    claims_b = grouped.loc[test, "sum"]
    n_b = grouped.loc[test, "count"]

    pooled = (claims_a + claims_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * ((1 / n_a) + (1 / n_b)))
    z_score = ((claims_a / n_a) - (claims_b / n_b)) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return {
        "test": "two-proportion z-test",
        "segment": segment_col,
        "control": control,
        "test_group": test,
        "control_rate": claims_a / n_a,
        "test_rate": claims_b / n_b,
        "statistic": z_score,
        "p_value": p_value,
    }


def welch_t_test(df, segment_col, control, test, metric="Margin"):
    """Compare a numerical KPI for two selected categories."""
    data = add_testing_metrics(df)
    a = data.loc[data[segment_col] == control, metric]
    b = data.loc[data[segment_col] == test, metric]
    statistic, p_value = stats.ttest_ind(
        a,
        b,
        equal_var=False,
        nan_policy="omit",
    )

    return {
        "test": "Welch t-test",
        "segment": segment_col,
        "metric": metric,
        "control": control,
        "test_group": test,
        "control_mean": a.mean(),
        "test_mean": b.mean(),
        "statistic": statistic,
        "p_value": p_value,
    }


def decision(p_value, alpha=0.05):
    """Return the null-hypothesis decision at the selected alpha."""
    if p_value < alpha:
        return "Reject H0"
    return "Fail to reject H0"
