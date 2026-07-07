# Transform raw features into better features.\
# Name -> Title
# Cabin -> Deck
# SibSp + Parch -> FamilySize
# FamilySize -> IsAlone
# Remove Name

from typing import Any
import pandas as pd

from src.config import RAW_COLUMNS_TO_REMOVE


def extract_title(df:pd.DataFrame) -> pd.DataFrame :
    """
    Extract passenger title from Name column.
    """
    df = df.copy()
    df['Title'] = (df['Name'].str.extract(r" ([A-Za-z]+)\.", expand=False))

    return df

def _create_family_size(df:pd.DataFrame) -> pd.DataFrame :
    """
    Create FamilySize feature.
    """
    df = df.copy()

    df['FamilySize'] = (df['SibSp'] + df['Parch'] +1)

    return df

def _drop_unused_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return df.drop(columns=columns)


def engineer_features(df:pd.DataFrame) -> pd.DataFrame :
    """
    Execute complete feature engineering pipeline.
    """
    df = extract_title(df)
    df = _create_family_size(df)
    df = _drop_unused_columns(df, RAW_COLUMNS_TO_REMOVE)
    return df

