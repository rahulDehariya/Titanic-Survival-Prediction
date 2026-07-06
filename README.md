notebooks/Titanic_Project.ipynb

Is for for:
- Understanding data
- Visualization
- Trying ideas

src/data_loader.py
Responsible for:
- Reading CSV files
- Returning DataFrames
Nothing else.

src/preprocessing.py
Responsible for:
- Missing values
- Encoding
- Scaling
- Train/Test preprocessing
No model training.

src/feature_engineering.py
Responsible for:
- Extract Title
- Create FamilySize
- Future engineered features

src/model.py

Responsible for:
- Creating models
- Training
- Saving models

src/evaluate.py
Responsible for:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

src/predict.py
Responsible for:
- Loading saved model
- Predicting new passengers


train.py
This becomes the orchestrator.
Load Data
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Split Data
      ↓
Train Model
      ↓
Evaluate
      ↓
Save Model
This file coordinates the entire workflow.