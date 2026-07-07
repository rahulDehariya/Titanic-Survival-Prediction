from pathlib import Path

import pandas as pd

from src.data_loader import load_data

def test_load_data_returns_dataframe():
    df = load_data("data/train.csv")
    assert isinstance(df, pd.DataFrame)
