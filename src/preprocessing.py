# This module should ONLY handle:
# Raw Features
# Missing Values
# Encoding
# Scaling
# Ready for Model
# Sex --> One-Hot Encoding
# Embarked --> One-Hot Encoding
# Title --> One-Hot Encoding
# Pclass --> One-Hot (for a general-purpose pipeline)

# Because these are nominal categorical features. There's no inherent ranking among the categories.
# Everything will be handled automatically by a Pipeline and ColumnTransformer.

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

NUMERICAL_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize"
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "Pclass"
]

# A pipeline is simply a sequence of transformations.
# Age ->SimpleImputer -> StandardScaler -> Output Each step receives the output of the previous step.
# imputer and scaler is the identifier
# Why StandardScaler?
# Because we want a preprocessing pipeline that works well across multiple models. Even though tree-based models don't require scaling, models like Logistic Regression, SVM, and KNN benefit from it.

def build_preprocessing() -> ColumnTransformer:
    """
    Build and return the preprocessing pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Categorical features Missing Values -> Most Frequent Value -> One Hot Encoding

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "scaler",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # Why handle_unknown="ignore"?
    # because your prediction will fail because the encoder has never seen "Prof".
    # with "Ignore" No crash, Pipleline still works

    # What ColumnTransformer - Apply different preprocessing pipelines to different columns.


    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )
    return column_transformer
