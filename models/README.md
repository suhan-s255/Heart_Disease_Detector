# Trained Models

This directory contains the trained machine learning models for heart disease prediction.

## Models

The following models are generated after running the HRLFM pipeline (`train_hrlfm_pipeline.py`):

1. **logistic_regression_model.pkl**: Logistic Regression model - interpretable linear model
2. **random_forest_model.pkl**: Random Forest Classifier model - robust ensemble approach
3. **hrlfm_model.pkl**: HRLFM (High-Resolution Logistic-Forest Model) - hybrid ensemble
4. **best_model.pkl**: The best performing model based on accuracy
5. **scaler.pkl**: RobustScaler for feature normalization
6. **feature_names.pkl**: List of selected feature names used by the models

## Additional Files

- **model_comparison.csv**: Performance metrics comparison across all models
- **confusion_matrix.png**: Confusion matrix visualization for the best model
- **roc_curve.png**: ROC curves for all models
- **feature_importance.png**: Top 20 most important features

## Usage

These models are automatically loaded by the Streamlit application (`app/streamlit_app.py`).

To use a model directly in Python:

```python
import joblib
import pandas as pd
import numpy as np

# Load the model and preprocessing objects
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare input data (must include all engineered features)
# See train_hrlfm_pipeline.py for feature engineering details
base_data = {
    'age': 55,
    'sex': 1,
    'cp': 2,
    'trestbps': 140,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalachh': 150,
    'exang': 0,
    'oldpeak': 2.0,
    'slope': 1,
    'ca': 0,
    'thal': 2
}

# Engineer features (same as training pipeline)
engineered_data = {}
engineered_data['age_chol_interaction'] = base_data['age'] * base_data['chol']
engineered_data['age_bp_interaction'] = base_data['age'] * base_data['trestbps']
# ... (add all engineered features as in training)

# Combine all features
all_features = {**base_data, **engineered_data}
input_df = pd.DataFrame([all_features])

# Select features used by model
input_df = input_df[feature_names]

# Scale features
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

print(f"Prediction: {'Disease' if prediction[0] == 1 else 'No Disease'}")
print(f"Disease Probability: {probability[0][1]:.2%}")
```

## Model Performance Metrics

The models are evaluated using comprehensive metrics:

### Metrics Explained

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
  - Proportion of correct predictions overall
  
- **Precision**: TP / (TP + FP)
  - Proportion of positive predictions that were correct
  - Important for reducing false alarms
  
- **Recall**: TP / (TP + FN)
  - Proportion of actual positives that were identified
  - Critical in medical diagnosis to avoid missing disease cases
  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - Harmonic mean balancing precision and recall
  
- **ROC-AUC**: Area under ROC curve
  - Measures ability to distinguish between classes
  - Values closer to 1.0 indicate better performance

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

## Pipeline Details

- **Training script**: `train_hrlfm_pipeline.py`
- **Dataset**: 1,888 patient records (80/20 train/test split)
- **Feature engineering**: 23 additional features created
- **Feature selection**: 36 most informative features selected
- **Cross-validation**: 5-fold stratified CV
- **Hyperparameter tuning**: RandomizedSearchCV
- **Class balancing**: SMOTE on training data
- **Feature scaling**: RobustScaler (less sensitive to outliers)

## Notes

- Models are saved using joblib for efficient serialization
- The scaler is applied to all features before prediction
- Feature names must match those used during training
- Test dataset is saved separately as `data/test_dataset.csv`
- To regenerate models, run: `python train_hrlfm_pipeline.py`
- Model files are excluded from git via `.gitignore`
