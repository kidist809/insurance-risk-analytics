# Insurance Risk Analytics & Predictive Modeling

This project analyzes motor-insurance policy and claim data for AlphaCare
Insurance Solutions. The goal is to identify low-risk customer and vehicle
segments, statistically test risk differences, and build reusable modeling code
for claim probability, claim severity, and risk-based premium indication.

> Data note: the local file in this repository is `data/insurance_data.csv`.
> It contains 10,000 Ethiopian motor-policy records from January 2024 to June
> 2025, which differs from the South African schema described in the challenge
> brief. The analysis and reports document this limitation transparently.

## Project Structure

- `notebooks/` - EDA, hypothesis testing, and modeling notebooks
- `src/` - reusable Python modules
- `reports/` - interim and final reports
- `data/` - data files tracked with DVC, not Git
- `tests/` - lightweight unit tests for reusable functions

## Reproduce the Environment

```bash
pip install -r requirements.txt
pytest
```

The GitHub Actions workflow in `.github/workflows/ci.yml` installs
dependencies, runs `flake8 src`, and then runs `pytest`.

## DVC Data Pipeline

DVC has been initialized and the raw dataset is tracked through
`data/insurance_data.csv.dvc`.

Configured local remote:

```bash
dvc remote add -d localstorage ../../dvc-storage
```

Typical reproduction commands:

```bash
dvc pull
dvc repro
```

The `dvc.yaml` file defines a `clean_data` stage that parses the raw dataset,
removes duplicate rows, and writes `data/insurance_data_clean.csv`.

## Final Workflow

1. Run EDA in `notebooks/01_eda.ipynb`.
2. Run statistical tests in `notebooks/02_hypothesis_testing.ipynb`.
3. Run severity and claim-probability modeling in `notebooks/03_modeling.ipynb`.
4. Review the business narrative in `reports/final_report.md`.
