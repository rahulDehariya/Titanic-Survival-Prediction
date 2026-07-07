from sklearn.ensemble import RandomForestClassifier
from models.model import build_model

def test_build_model_returns_random_forest():
    model = build_model()

    assert isinstance(model, RandomForestClassifier)