RAW_COLUMNS_TO_REMOVE = [
    "Name",
]
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "titanic_model.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
