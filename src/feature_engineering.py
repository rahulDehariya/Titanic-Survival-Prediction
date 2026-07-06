# Transform raw features into better features.\
# Name -> Title
# Cabin -> Deck
# SibSp + Parch -> FamilySize
# FamilySize -> IsAlone
# Remove Name

from typing import Any
import pandas as pd

def extractTitle(df:pd.DataFrame) -> pd.DataFrame :
    """
    Extract passenger title from Name column.
    """
    df = df.copy()
    df['Title'] = (df['Name'].str.extract(r" ([A-Za-z]+)\.", expand=False))

    return df

def createFamilySize(df:pd.DataFrame) -> pd.DataFrame :
    """
    Create FamilySize feature.
    """
    df = df.copy()

    df['FamilySize'] = (df['SibSp'] + df['Parch'] +1)

    return df

def dropUnusedColumn(df, column) :
    """
    Drop columns that are no longer needed.
    """
    return df.drop(column=column)


def engineeringFeature(df:pd.DataFrame) -> pd.DataFrame :
    """
    Execute complete feature engineering pipeline.
    """
    extractTitle(df)
    createFamilySize(df)
    dropUnusedColumn(df)
    return df

