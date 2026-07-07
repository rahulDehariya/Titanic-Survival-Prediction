from sklearn.compose import ColumnTransformer
from src.preprocessing import build_preprocessing

import pandas as pd

def test_build_preprocessing_returns_column_transformer():
    preprocessing = build_preprocessing()

    assert isinstance(preprocessing, ColumnTransformer)
