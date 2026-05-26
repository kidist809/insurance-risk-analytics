import pandas as pd

from src.eda_utils import add_insurance_metrics
from src.hypothesis_tests import decision
from src.modeling import risk_based_premium


def test_add_insurance_metrics():
    df = pd.DataFrame(
        {
            "TotalPremium": [1000, 2000],
            "TotalClaims": [0, 500],
        }
    )

    result = add_insurance_metrics(df)

    assert result["Margin"].tolist() == [1000, 1500]
    assert result["ClaimFrequencyFlag"].tolist() == [0, 1]


def test_decision():
    assert decision(0.01) == "Reject H0"
    assert decision(0.50) == "Fail to reject H0"


def test_risk_based_premium():
    assert risk_based_premium(0.2, 10000, 0.1, 0.15) == 2500
