import pandas as pd

from src.feature_engineering import engineer_features


def test_family_size():
    df = pd.DataFrame(
        {
            "Name": ["Mr. Rahul"],
            "SibSp": [2],
            "Parch": [1],
        }
    )
    result = engineer_features(df)

    assert result.loc[0, "FamilySize"] == 4


def test_title_extraction():
    df = pd.DataFrame(
        {
            "Name": ["Mr. Rahul"],
            "SibSp": [0],
            "Parch": [0],
        }
    )

    result = engineer_features(df)
    result.loc[0, "Title"]


def test_name_column_removed():
    df = pd.DataFrame(
        {
            "Name": ["Mr. Rahul"],
            "SibSp": [0],
            "Parch": [0],
        }
    )

    result = engineer_features(df)

    assert "Name" not in result.columns
