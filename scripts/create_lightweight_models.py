"""Create lightweight pre-trained models for the Streamlit app.
This script trains a single RandomForest (n_estimators=100, n_jobs=1) with minimal CV
and saves the model, scaler, and feature names to `models/`.
It's intentionally conservative to avoid heavy CPU use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


def engineer_features(df):
    df_engineered = df.copy()
    # Interaction features
    if 'age' in df.columns and 'chol' in df.columns:
        df_engineered['age_chol_interaction'] = df['age'] * df['chol']
    if 'age' in df.columns and 'trestbps' in df.columns:
        df_engineered['age_bp_interaction'] = df['age'] * df['trestbps']
    if 'trestbps' in df.columns and 'chol' in df.columns:
        df_engineered['bp_chol_ratio'] = df['trestbps'] / (df['chol'] + 1)
    if 'thalachh' in df.columns and 'age' in df.columns:
        df_engineered['heart_rate_reserve'] = df['thalachh'] - df['age']
    if 'oldpeak' in df.columns and 'slope' in df.columns:
        df_engineered['st_depression_severity'] = df['oldpeak'] * (df['slope'] + 1)
    if 'cp' in df.columns and 'exang' in df.columns:
        df_engineered['chest_pain_exercise_risk'] = df['cp'] * (df['exang'] + 1)

    # Age groups
    if 'age' in df.columns:
        df_engineered['age_group'] = pd.cut(
            df['age'], bins=[0, 40, 55, 70, 120], labels=[0,1,2,3]
        ).astype(int)

    # Chol categories
    if 'chol' in df.columns:
        df_engineered['chol_category'] = pd.cut(
            df['chol'], bins=[0, 200, 240, 600], labels=[0,1,2]
        ).astype(int)

    # Polynomial features for key columns
    key_features = [f for f in ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak'] if f in df_engineered.columns]
    if key_features:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_data = poly.fit_transform(df_engineered[key_features])
        poly_feature_names = poly.get_feature_names_out(key_features)
        for i, name in enumerate(poly_feature_names):
            if name not in key_features:
                df_engineered[f'poly_{name}'] = poly_data[:, i]

    return df_engineered


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / 'data' / 'cleaned_merged_heart_dataset.csv'
    models_dir = root / 'models'
    models_dir.mkdir(exist_ok=True)

    print('Loading data...')
    df = pd.read_csv(data_path)

    print('Engineering features (lightweight)...')
    df_eng = engineer_features(df)

    # Prepare X, y
    X = df_eng.drop(columns=['target'])
    y = df_eng['target']

    # Simple train/test split (we won't perform heavy CV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit a RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a RandomForest (single-threaded)
    print('Training RandomForest (n_estimators=100, n_jobs=1)...')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train_scaled, y_train)

    # Save model and preprocessing
    print('Saving model and preprocessing artifacts...')
    with open(models_dir / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open(models_dir / 'best_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # Save feature names (list)
    feature_names = list(X.columns)
    with open(models_dir / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # Also save a tiny model_comparison.csv
    comp = {
        'Model': ['Random Forest'],
        'Accuracy': [round(rf.score(X_test_scaled, y_test), 4)],
        'Precision': [0],
        'Recall': [0],
        'F1-Score': [0],
        'ROC-AUC': [0]
    }
    pd.DataFrame(comp).to_csv(models_dir / 'model_comparison.csv', index=False)

    print('Done. Files written to models/:', list(models_dir.iterdir()))


if __name__ == '__main__':
    main()
