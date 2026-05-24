import pandas as pd

def load_data(path):
    """
    Load insurance dataset
    """
    df = pd.read_csv(path)
    return df 
