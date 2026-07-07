import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing import build_preprocessing


def test_build_preprocessing_returns_column_transformer():
    preprocessing = build_preprocessing()

    assert isinstance(preprocessing, ColumnTransformer)
