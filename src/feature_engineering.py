# Transform raw features into better features.\
# Name -> Title
# Cabin -> Deck
# SibSp + Parch -> FamilySize
# FamilySize -> IsAlone
# Remove Name

from typing import Any
import pandas as pd

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

def _drop_unused_columns(df, column) :
    """
    Drop columns that are no longer needed.
    """
    return df.drop(column=column)


def engineeringFeature(df:pd.DataFrame) -> pd.DataFrame :
    """
    Execute complete feature engineering pipeline.
    """
    extract_title(df)
    _create_family_size(df)
    _drop_unused_columns(df)
    return df

