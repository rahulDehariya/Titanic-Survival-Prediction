import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.model import build_model
from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE, TRAIN_DATA_PATH
from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessing


def main() -> None:
    # Load dataset
    df = load_data(TRAIN_DATA_PATH)

    # Feature Engineering
    df = engineer_features(df)

    # Separate features and target
    X = df.drop(columns="Survived")
    Y = df["Survived"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # build component
    preprocessor = build_preprocessing()
    model = build_model()

    pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation
    metrics = evaluate_model(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    # Compare Actual vs Predicted
    prediction_df = X_test.copy()

    prediction_df["Actual"] = y_test.values
    prediction_df["Predected"] = y_pred

    # Predict New Passengers
    new_passenger = pd.DataFrame(
        [
            {
                "Pclass": 1,
                "Sex": "female",
                "Age": 30,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 80,
                "Embarked": "S",
                "Title": "Miss",
                "FamilySize": 1,
            }
        ]
    )

    prediction = pipeline.predict(new_passenger)
    # We save the model.
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    main()
