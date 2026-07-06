# Why RandonForest
# - Strong baseline
# - Handles nonlinear relationships
# - Doesn't require heavy feature scaling
# - Robust on tabular datasets

from sklearn.ensemble import RandomForestClassifier


def build_model() -> RandomForestClassifier:
    """
    Build and return the machine learning model.
    """

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
    )

    return model