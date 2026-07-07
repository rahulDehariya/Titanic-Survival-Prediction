# Load dataset from disk and return a DataFrame.

from pathlib import Path

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(file_path)
