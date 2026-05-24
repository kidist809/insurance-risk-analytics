 
# Interim Report: Insurance Risk Analytics and Data Versioning

## 1. Executive Summary

AlphaCare Insurance Solutions (ACIS) wants to move toward evidence-driven motor insurance pricing and marketing. The interim phase of this project focused on setting up a reproducible analytics repository, performing exploratory data analysis (EDA), and beginning data version control with DVC.

The current analysis uses the local dataset available in `data/insurance_data.csv`. It contains 10,000 policy records and 21 fields covering customer demographics, vehicle attributes, premiums, claims, cover type, and transaction dates. One important data caveat is that the local file differs from the original challenge description: it uses a compact schema with fields such as `CustomerID`, `Age`, `Province`, `VehicleType`, `AnnualPremium`, `TotalPremium`, `TotalClaims`, `AutoMake`, and `TransactionDate`; it also contains Ethiopian province names and dates from January 2024 to June 2025 rather than the described South African data from February 2014 to August 2015. The findings below therefore describe the dataset currently available in the repository.

Key interim findings:

- The portfolio has a total premium of 24,881,279 and total claims of 13,141,885.
- Overall loss ratio is 0.528, meaning claims consume about 52.8% of earned premium.
- Aggregate margin is positive at 11,739,394.
- Claim frequency is 15.35%, and average claim severity among claim policies is 8,561.49.
- Somali has the highest provincial loss ratio at 0.612, while Amhara has the lowest at 0.478.
- Luxury vehicles show the highest loss ratio at 0.842, making them a clear priority for deeper risk testing.
- Gender differences are very small in the current EDA: Female loss ratio is 0.529 and Male loss ratio is 0.528.

## 2. Business Understanding

The business goal is to identify lower-risk customer and vehicle segments that ACIS can target with more competitive premiums, while also detecting high-risk segments that may require pricing adjustments. The two main profitability metrics used in this interim analysis are:

- Loss Ratio = TotalClaims / TotalPremium
- Margin = TotalPremium - TotalClaims

A lower loss ratio suggests a more profitable or lower-risk segment. A higher loss ratio suggests that claims are consuming a larger share of premium and may require underwriting, pricing, or product design attention.

At this stage, the EDA is intended to guide later hypothesis testing and predictive modeling. The strongest interim signals are geographic variation and vehicle-type variation, especially the higher observed risk for Somali province and luxury vehicles.

## 3. Repository and Engineering Setup

The project has been organized into a reproducible Python repository with the following completed elements:

- `notebooks/01_eda.ipynb` for exploratory data analysis.
- `src/data_loader.py` with a reusable `load_data()` function.
- `src/eda_utils.py` with reusable helper functions for missing-value checks, summaries, data types, loss ratio, and margin creation.
- `tests/test_basic.py` with a basic test scaffold.
- `.github/workflows/ci.yml` configured to install dependencies, run `flake8 src`, and run `pytest` on every push.
- `requirements.txt` listing core analytics and modeling dependencies, including pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, pytest, flake8, dvc, xgboost, and shap.

The Git history shows multiple descriptive commits for project setup, CI, EDA, and DVC tracking. The current local branch is `task-1`, ahead of `origin/task-1` by two commits, with DVC-related changes present locally.

## 4. Data Overview

The local dataset has:

- Rows: 10,000
- Columns: 21
- Date range: 2024-01-01 to 2025-06-29
- Missing values: none detected in the current dataset

Main fields used in the interim EDA include:

- Customer and demographic fields: `CustomerID`, `Age`, `Gender`, `Province`, `AnnualIncome`
- Vehicle fields: `VehicleType`, `AutoMake`, `VehicleModel`, `CustomValueEstimate`
- Risk and policy fields: `RiskScore`, `Deductible`, `NCD`, `PastClaims`, `CoverType`
- Claim and premium fields: `Claimed`, `ClaimAmount`, `TotalPremium`, `TotalClaims`
- Time and location fields: `ZipCode`, `TransactionDate`

Because no missing values were found, no imputation was applied during the interim EDA. A cleaned export step was attempted in the notebook, but it failed because the notebook attempted to write to `data/insurance_data_clean.csv` from a working directory where `data` did not exist. This should be corrected in the next iteration by using a repository-root-relative path or resolving paths with `pathlib`.

## 5. Exploratory Data Analysis Findings

### 5.1 Portfolio-Level Risk and Profitability

The portfolio is profitable at aggregate level:

| Metric | Value |
|---|---:|
| Total Premium | 24,881,279 |
| Total Claims | 13,141,885 |
| Overall Loss Ratio | 0.528 |
| Aggregate Margin | 11,739,394 |
| Claim Frequency | 0.1535 |
| Claim Severity | 8,561.49 |

The overall loss ratio of 52.8% suggests that the dataset contains room for profitable segmentation: not all premium is being consumed by claims, and several subgroups show meaningfully different loss ratios.

### 5.2 Financial Variable Distributions

Summary statistics show that claim amounts are highly skewed. Most policies have zero claims, while a small number of policies produce large claim amounts.

| Metric | Min | Mean | Median | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| TotalPremium | 951 | 2,488.13 | 2,307 | 2,676 | 4,348 | 5,105 |
| TotalClaims | 0 | 1,314.19 | 0 | 0 | 9,200 | 49,623 |
| AnnualIncome | 17,202 | 79,201.97 | 72,933 | 95,090 | 140,966 | 376,916 |
| CustomValueEstimate | 5,022 | 35,640.60 | 28,520 | 46,720 | 88,677 | 134,914 |
| RiskScore | 15 | 58.14 | 57 | 67 | 85 | 95 |
| Age | 18 | 46.68 | 46 | 61 | 73 | 75 |

The median `TotalClaims` is zero and the 75th percentile is also zero, confirming that most policies did not generate claims. The maximum claim value of 49,623 is much higher than the 95th percentile of 9,200, so outlier-aware analysis will be important for modeling.

### 5.3 Geographic Risk Patterns

Loss ratio varies by province:

| Province | Policies | Total Premium | Total Claims | Loss Ratio |
|---|---:|---:|---:|---:|
| Somali | 1,184 | 2,984,984 | 1,826,593 | 0.612 |
| Oromia | 2,446 | 6,069,663 | 3,261,061 | 0.537 |
| Tigray | 804 | 1,990,692 | 1,047,136 | 0.526 |
| Addis Ababa | 3,567 | 8,907,374 | 4,653,210 | 0.522 |
| Amhara | 1,999 | 4,928,566 | 2,353,885 | 0.478 |

Somali has the highest observed loss ratio, while Amhara has the lowest. This is a useful first signal for later hypothesis testing. However, this finding should not yet be treated as causal because province may be confounded by vehicle mix, cover type, customer risk score, or income.

### 5.4 Vehicle Type Risk Patterns

Vehicle type shows a clearer risk separation:

| Vehicle Type | Policies | Loss Ratio | Average Claims |
|---|---:|---:|---:|
| Luxury | 972 | 0.842 | 3,672.02 |
| SUV | 3,000 | 0.564 | 1,363.56 |
| Hatchback | 2,036 | 0.420 | 935.28 |
| Sedan | 3,992 | 0.404 | 896.24 |

Luxury vehicles have the highest loss ratio and average claims. Sedans and hatchbacks appear lower risk in this interim analysis. This suggests a potential pricing opportunity: luxury vehicles may need tighter underwriting or higher risk loading, while sedans and hatchbacks may contain lower-risk segments for marketing campaigns.

### 5.5 Gender Risk Patterns

Gender does not show a meaningful difference at the EDA stage:

| Gender | Policies | Loss Ratio | Claim Frequency |
|---|---:|---:|---:|
| Female | 5,138 | 0.529 | 0.154 |
| Male | 4,862 | 0.528 | 0.153 |

The loss ratios and claim frequencies are nearly identical. Based on EDA alone, there is no strong business case for gender-based pricing differences in the current dataset. This should be confirmed formally in Task 3 using a statistical hypothesis test.

### 5.6 Vehicle Make Patterns

Average claim amounts also vary by auto make:

| Auto Make | Policies | Average Claims | Claim Frequency |
|---|---:|---:|---:|
| Mercedes-Benz | 317 | 3,787.01 | 0.281 |
| BMW | 339 | 3,362.89 | 0.260 |
| Toyota | 3,281 | 1,372.52 | 0.161 |
| Suzuki | 1,708 | 1,122.03 | 0.138 |
| Hyundai | 3,076 | 1,042.74 | 0.140 |
| Lifan | 1,279 | 918.10 | 0.128 |

Mercedes-Benz and BMW show higher average claims and claim frequency than other makes. This aligns with the vehicle-type finding that luxury vehicles may be a higher-risk segment.

### 5.7 Temporal Patterns

The data covers 18 transaction months from January 2024 through June 2025. Average monthly claims fluctuate, with May 2024 showing the highest average claims in this period and April 2025 showing the lowest.

Examples:

- Highest observed monthly average claims: May 2024 at 1,636.65
- Lowest observed monthly average claims: April 2025 at 937.88
- Claim frequency ranges from 0.119 in April 2025 to 0.180 in February 2025

There is no clear monotonic upward or downward trend from the interim EDA alone. The next step should separate claim frequency and severity over time to check whether seasonal or portfolio-mix changes are driving the fluctuations.

## 6. Visual Analysis Completed

The EDA notebook includes several visual checks:

- Scatter plot of `TotalPremium` versus `TotalClaims`
- Histogram of `TotalPremium`
- Histogram of `TotalClaims`
- Box plot for `TotalClaims`
- Box plot for `CustomValueEstimate`
- Bar chart of loss ratio by province
- Bar chart of loss ratio by gender
- Bar chart of loss ratio by vehicle type
- Monthly claim trend plot

These plots support the main interim findings: claims are zero-inflated and skewed, vehicle type has strong risk separation, and province appears to have a measurable association with portfolio loss ratio.

## 7. DVC Setup

DVC has been initialized in the repository. The following DVC artifacts are present:

- `.dvc/`
- `.dvcignore`
- `.dvc/config`
- `data/insurance_data.csv.dvc`

The dataset is tracked through DVC with the following metadata:

- DVC output path: `insurance_data.csv`
- MD5: `1846c9c80019e93300c46191738efbe9`
- Size: 1,342,065 bytes

The configured default DVC remote is:

```ini
[core]
    remote = localstorage
[remote "localstorage"]
    url = ../../dvc-storage
```

This establishes the foundation for reproducible data tracking. However, the local environment used for this report does not currently expose the `dvc` command, so I could not verify `dvc status` or `dvc push` from the shell. Also, only the raw dataset DVC file is currently visible; a second cleaned dataset version is not yet present in `data/`.

## 8. Current Limitations

The main limitations at interim stage are:

- The available dataset does not match the full challenge schema or geography described in the brief.
- The EDA is descriptive only; statistical significance has not yet been tested.
- Province, vehicle type, and auto make effects may be confounded by other variables.
- Claim data is zero-inflated and skewed, so simple averages can be influenced by outliers.
- The cleaned dataset export in the notebook failed due to a path issue.
- `pytest` and `dvc` were not available in the current shell, so local verification of tests and DVC status could not be completed during report writing.
- The CI workflow exists, but it should be confirmed on GitHub after pushing the latest commits.

## 9. Interim Recommendations

Based on the current EDA, ACIS should prioritize the following:

1. Investigate luxury vehicles as a high-risk segment.

Luxury vehicles have a loss ratio of 0.842, much higher than sedans at 0.404 and hatchbacks at 0.420. This segment should be tested formally and may require higher premiums, stricter underwriting, or differentiated excess structures.

2. Explore lower-risk marketing segments among sedans and hatchbacks.

Sedans and hatchbacks currently show the lowest vehicle-type loss ratios. These may be good candidates for acquisition campaigns if later testing confirms they remain low risk after controlling for customer and cover attributes.

3. Treat Somali province as a risk-pricing investigation area.

Somali has the highest provincial loss ratio at 0.612. Before changing pricing, ACIS should test whether this difference remains significant after comparing similar customers, vehicles, and plan types.

4. Avoid gender-based conclusions at this stage.

Gender differences are negligible in the current EDA. Formal testing is still needed, but the descriptive evidence does not support gender as a major risk differentiator in this dataset.

5. Build separate frequency and severity models.

Because most policies have no claims, a two-part modeling strategy is appropriate: first predict probability of claim, then predict claim severity for policies where a claim occurs. This aligns with the final risk-based pricing objective.

## 10. Next Steps

For the final submission, the next work should focus on:

- Fixing the notebook path issue and generating a clean data version.
- Completing DVC verification and pushing tracked data to the configured local remote.
- Adding a README section explaining how to reproduce the DVC data pipeline.
- Running Task 3 hypothesis tests for province, zip code, margin by zip code, and gender.
- Building Task 4 models for claim severity and claim probability.
- Comparing Linear Regression, Random Forest, and XGBoost models.
- Adding SHAP or LIME interpretation for the best-performing model.
- Converting the final report into a polished business-facing narrative with EDA, hypothesis testing, modeling, recommendations, limitations, and next steps.
