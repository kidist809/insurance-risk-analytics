from pathlib import Path

import pandas as pd


def load_data(path="data/insurance_data.csv"):
    """Load the insurance dataset and parse known date columns."""
    data_path = Path(path)
    df = pd.read_csv(data_path)

    date_columns = ["TransactionDate", "TransactionMonth", "VehicleIntroDate"]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df
