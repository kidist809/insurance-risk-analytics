# Final Report: Insurance Risk Analytics and Predictive Modeling

## Leadership Overview

AlphaCare Insurance Solutions wants to move from intuition-led motor-insurance
pricing toward evidence-driven pricing and marketing. This project built a
reproducible analytics foundation, explored portfolio risk, tested whether
important customer and location segments have statistically different risk, and
implemented reusable modeling code for claim severity and claim probability.

The available local dataset contains 10,000 policy records from January 2024 to
June 2025. It does not match the original South African challenge schema, so the
findings below describe the repository dataset rather than the external brief.
This limitation matters, but the analytical workflow remains aligned with the
business objective.

## Data and Metrics

The analysis uses two core insurance metrics:

- Loss Ratio = TotalClaims / TotalPremium
- Margin = TotalPremium - TotalClaims

The portfolio generated total premium of 24,881,279 and total claims of
13,141,885. The overall loss ratio is 0.528, meaning claims consume about 52.8%
of premium. Claim frequency is 15.35%, and average claim severity among policies
with claims is 8,561.49.

## Exploratory Insights

The portfolio is profitable overall, but risk is not evenly distributed.

| Segment | Key Finding |
|---|---|
| Province | Somali has the highest loss ratio at 0.612; Amhara has the lowest at 0.478. |
| Vehicle Type | Luxury vehicles have the highest loss ratio at 0.842, far above sedans at 0.404. |
| Gender | Female and Male loss ratios are nearly identical at 0.529 and 0.528. |
| Auto Make | Mercedes-Benz and BMW have the highest average claims and claim frequencies. |
| Time | Monthly claim frequency fluctuates, with no clear monotonic trend. |

Claims are zero-inflated and skewed. The median `TotalClaims` is 0, while the
maximum claim is 49,623. This supports a two-part modeling approach: first
predict whether a claim occurs, then estimate severity for claim policies.

## Hypothesis Testing

The tests use claim frequency for risk hypotheses and margin for profitability.
For zip-code testing, `30002` is used as the lower-risk comparison group and
`40002` as the higher-risk comparison group based on observed claim frequency.

| Hypothesis | KPI | Test | p-value | Decision |
|---|---|---|---:|---|
| No risk difference across provinces | Claim frequency | Chi-square | 0.075 approx. | Fail to reject H0 |
| No risk difference between zip codes 30002 and 40002 | Claim frequency | Two-proportion z-test | 0.00016 | Reject H0 |
| No margin difference between zip codes 30002 and 40002 | Margin | Welch t-test | 0.01797 | Reject H0 |
| No risk difference between Women and Men | Claim frequency | Two-proportion z-test | 0.94173 | Fail to reject H0 |

Business interpretation:

- Zip code has a statistically meaningful relationship with claim frequency in
  the selected comparison. Zip code `40002` has a claim frequency of 20.53%,
  compared with 10.10% for zip code `30002`.
- Average margin is also significantly different between these two zip codes.
  Zip code `30002` has a higher average margin, so it is a stronger candidate
  for lower-risk marketing campaigns.
- Gender does not show evidence of a meaningful risk difference in this dataset.
  ACIS should avoid gender-based pricing changes from this evidence.
- Province remains commercially important because Somali shows the highest loss
  ratio, but the claim-frequency test does not cross the 5% significance
  threshold in the current data.

## Modeling Approach

The modeling code in `src/modeling.py` implements:

- Claim severity prediction for policies where `TotalClaims > 0`
- Claim probability classification using a binary `ClaimFlag`
- Mixed numeric/categorical preprocessing with imputation and one-hot encoding
- Linear Regression, Random Forest, and XGBoost for severity where available
- Logistic Regression and Random Forest for claim probability
- A risk-based premium formula:

```text
Premium = P(claim) x Predicted Severity x (1 + Expense Loading + Profit Margin)
```

This design matches the insurance data shape better than a single model because
most policies have no claim while a smaller number produce large losses.

## Recommendations

1. Prioritize luxury vehicles for pricing review.
Luxury vehicles have the highest observed loss ratio. ACIS should test higher
risk loadings, stricter underwriting, or adjusted deductibles for this segment.

2. Use sedans, hatchbacks, and lower-risk zip codes for growth campaigns.
Sedans, hatchbacks, and zip code `30002` show stronger profitability signals.
These segments are candidates for targeted acquisition offers if model outputs
confirm lower expected loss.

3. Treat Somali province as a risk investigation area, not an immediate causal
pricing rule.
The descriptive loss ratio is high, but province may be confounded by vehicle
mix, cover type, risk score, and income.

4. Do not make gender-based pricing changes.
The observed Female and Male claim frequencies are almost identical, and the
hypothesis test fails to reject the null hypothesis.

5. Adopt the two-part pricing model.
ACIS should combine claim probability and claim severity predictions to produce
more stable premium indications than a direct average-claim approach.

## Reproducibility and Limitations

DVC is initialized and the raw dataset is tracked with
`data/insurance_data.csv.dvc`. A `dvc.yaml` stage has been added for producing a
clean data version. The local shell used for this report did not expose working
`python`, `pytest`, or `dvc` commands, so the notebooks and tests should be run
again after activating a proper Python environment.

The largest limitation is data mismatch: the local dataset differs from the
challenge description in geography, fields, and date range. Results should
therefore be presented as evidence from the available repository dataset, not as
definitive conclusions about the South African ACIS portfolio.

## Next Steps

- Run `pytest`, `dvc repro`, and all notebooks in a clean virtual environment.
- Execute SHAP on the best severity model and document the top features.
- Validate whether provincial differences remain after controlling for vehicle
  type, cover type, risk score, and income.
- Push the latest branches and merge Tasks 1-4 into `main` through pull
  requests for the final GitHub submission.
