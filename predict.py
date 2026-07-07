from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH, SUBMISSION_PATH, TEST_DATA_PATH
from src.data_loader import load_data
from src.feature_engineering import engineer_features

MODEL_PATH = Path(MODEL_PATH)


def main():
    # Load trained pipeline
    pipeline = joblib.load(MODEL_PATH)

    # Load test dataset
    test_df = load_data(TEST_DATA_PATH)

    # Save PassengerId before feature engineering
    passenger_ids = test_df["PassengerId"]

    # Apply feature engineering
    test_df = engineer_features(test_df)

    # predict survival
    prediction = pipeline.predict(test_df)

    # Create submission DataFrame

    submission = pd.DataFrame({"PassenderId": passenger_ids, "Survived": prediction})

    # Save submission
    submission.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
