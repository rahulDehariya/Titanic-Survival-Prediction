# Titanic Survival Prediction

A production-inspired Machine Learning project that predicts passenger survival on the Titanic using a modular, reusable architecture instead of a notebook-only workflow.

---

## Project Overview

The objective of this project is to build a binary classification model that predicts whether a passenger survived the Titanic disaster based on demographic and travel information.

Unlike many beginner projects that place the complete workflow inside a Jupyter notebook, this project separates concerns into reusable modules for feature engineering, preprocessing, model training, evaluation, and prediction.

---

## Business Problem

Shipping companies and insurers often need to analyze passenger risk and understand which factors influence survival during emergencies.

The goal of this project is to build a machine learning model capable of predicting passenger survival from historical passenger data.

**Target Variable**

* `Survived`

  * `0` → Did Not Survive
  * `1` → Survived

---

## Dataset

The project uses the Titanic dataset.

### Features

* PassengerId
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

---

## Project Structure

```text
Titanic Survival Prediction/

├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── model.py
│   └── titanic_model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── outputs/
│   └── submission.csv
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── evaluate.py
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Machine Learning Workflow

```text
Load Dataset
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Preprocessing
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Save Pipeline
      │
      ▼
Prediction
```

---

## Feature Engineering

The following features were engineered:

### Title Extraction

Passenger titles were extracted from the `Name` column.

Examples:

* Mr
* Mrs
* Miss
* Master

---

### Family Size

A new feature was created:

```
FamilySize = SibSp + Parch + 1
```

This represents the total number of family members traveling together.

---

## Preprocessing

### Numerical Features

* Median Imputation
* Standard Scaling

### Categorical Features

* Most Frequent Imputation
* One-Hot Encoding

The preprocessing workflow is implemented using:

* Pipeline
* ColumnTransformer

which helps prevent data leakage and keeps preprocessing reusable.

---

## Model

Current baseline model:

* Random Forest Classifier

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Baseline Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8324 |
| Precision | 0.7973 |
| Recall    | 0.7973 |
| F1 Score  | 0.7973 |
| ROC-AUC   | 0.8272 |

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

This will:

* Train the model
* Evaluate performance
* Save the trained pipeline as:

```
models/titanic_model.pkl
```

---

### Generate Predictions

```bash
python predict.py
```

This creates:

```
outputs/submission.csv
```

ready for Kaggle submission.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## Software Engineering Practices

This project follows several software engineering principles:

* Modular project structure
* Separation of concerns
* Reusable feature engineering
* Reusable preprocessing pipeline
* Type hints
* Clean function design
* Production-style training and prediction workflow

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
* Model comparison (Logistic Regression, XGBoost, LightGBM)
* Feature importance visualization
* Confusion matrix and ROC curve visualization
* Configuration management
* Logging
* Unit tests
* CI/CD pipeline
* Docker support
* REST API deployment

---

## Author

**Rahul Dehariya**

Software Engineer transitioning into Machine Learning and Data Science.

Passionate about building production-ready Machine Learning solutions using clean engineering practices.
