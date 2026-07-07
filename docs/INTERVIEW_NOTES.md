# Titanic Survival Prediction – Project Q&A

## 1. What is the objective of this project?

The objective is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, fare, and family details.

The problem is a **binary classification** problem because the target variable (`Survived`) has only two possible values:

* 0 → Did Not Survive
* 1 → Survived

---

## 2. Why did you choose Random Forest?

Random Forest was selected as the baseline model because it:

* Handles nonlinear relationships well.
* Is robust to noisy data.
* Works well on tabular datasets.
* Requires minimal feature scaling.
* Provides strong baseline performance with little hyperparameter tuning.

The project was designed so that the model can easily be replaced with Logistic Regression, XGBoost, or LightGBM without changing the training workflow.

---

## 3. Why did you create a preprocessing pipeline?

The preprocessing pipeline ensures that the exact same preprocessing steps are applied during both training and prediction.

It performs:

* Missing value imputation
* One-Hot Encoding
* Feature Scaling

Using a scikit-learn Pipeline reduces code duplication and helps prevent inconsistencies between training and inference.

---

## 4. Why did you use ColumnTransformer?

Different columns require different preprocessing techniques.

Numerical features need:

* Median Imputation
* Standard Scaling

Categorical features need:

* Most Frequent Imputation
* One-Hot Encoding

ColumnTransformer allows different preprocessing pipelines to be applied to different feature groups within a single reusable workflow.

---

## 5. What feature engineering did you perform?

Two new features were created.

### Title

The passenger title (Mr, Mrs, Miss, Master, etc.) was extracted from the Name column because it captures useful demographic information.

### FamilySize

A new feature was created:

FamilySize = SibSp + Parch + 1

This represents the total number of family members travelling together.

---

## 6. Why did you save the entire pipeline instead of only the model?

The saved pipeline contains:

* Feature preprocessing
* Encoding
* Scaling
* Trained machine learning model

This ensures that new data is transformed exactly the same way as the training data before making predictions.

---

## 7. Why did you separate train.py and predict.py?

Each file has a single responsibility.

**train.py**

* Load training data
* Train the model
* Evaluate performance
* Save the trained pipeline

**predict.py**

* Load the saved pipeline
* Predict survival for unseen passengers
* Generate the submission file

Separating training and inference makes the project easier to maintain and closer to production practices.

---

## 8. How did you evaluate the model?

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

These metrics provide a more complete view of model performance than accuracy alone.

---

## 9. What software engineering practices did you follow?

The project follows several engineering best practices:

* Modular project structure
* Single Responsibility Principle
* Configuration management
* Reusable preprocessing pipeline
* Unit testing with pytest
* Clean function design
* Type hints
* Production-style project organization

---

## 10. What would you improve next?

If this project were developed further, I would add:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Model comparison with Logistic Regression, XGBoost, and LightGBM
* Feature importance visualization
* Confusion Matrix and ROC Curve
* Cross-validation
* Logging to files
* Experiment tracking (e.g., MLflow)
* CI/CD pipeline for automated testing

---

## Key Learning

This project helped me understand that building a machine learning model is only one part of the workflow. A production-ready ML project also requires clean architecture, reusable preprocessing, proper testing, model persistence, and a separate inference pipeline.
