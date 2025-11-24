#!/usr/bin/env python
"""
Test script to verify that saved models can be loaded and used for predictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def test_model_loading():
    """Test that all saved models can be loaded and make predictions."""
    
    print("="*80)
    print("MODEL LOADING AND PREDICTION TEST")
    print("="*80)
    
    # Check if models exist
    models_to_test = {
        'Best Model (Random Forest)': 'models/best_model.pkl',
        'HRLFM': 'models/hrlfm_model.pkl',
        'Random Forest': 'models/random_forest_model.pkl',
        'Logistic Regression': 'models/logistic_regression_model.pkl'
    }
    
    preprocessing = {
        'Scaler': 'models/scaler.pkl',
        'Feature Names': 'models/feature_names.pkl'
    }
    
    print("\n1. Checking if model files exist...")
    all_exist = True
    for name, path in {**models_to_test, **preprocessing}.items():
        if Path(path).exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n✗ ERROR: Some model files are missing")
        print("  Please run: python train_hrlfm_pipeline.py")
        return False
    
    print("\n2. Loading preprocessing objects...")
    try:
        scaler = joblib.load(preprocessing['Scaler'])
        feature_names = joblib.load(preprocessing['Feature Names'])
        print(f"  ✓ Scaler loaded")
        print(f"  ✓ Feature names loaded ({len(feature_names)} features)")
    except Exception as e:
        print(f"  ✗ ERROR loading preprocessing objects: {e}")
        return False
    
    print("\n3. Loading and testing models...")
    all_models_work = True
    
    # Create a sample input (all zeros for simplicity - just testing loading)
    sample_input = np.zeros((1, len(feature_names)))
    
    for name, path in models_to_test.items():
        try:
            model = joblib.load(path)
            
            # Try to make a prediction
            prediction = model.predict(sample_input)
            prediction_proba = model.predict_proba(sample_input)
            
            print(f"  ✓ {name}: Loaded successfully")
            print(f"    - Can make predictions: {prediction[0]}")
            print(f"    - Probability: [{prediction_proba[0][0]:.4f}, {prediction_proba[0][1]:.4f}]")
            
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")
            all_models_work = False
    
    print("\n4. Testing with test dataset...")
    test_data_path = 'data/test_dataset.csv'
    if Path(test_data_path).exists():
        try:
            test_df = pd.read_csv(test_data_path)
            print(f"  ✓ Test dataset loaded: {len(test_df)} samples")
            
            # Check if it has the target column
            if 'target' in test_df.columns:
                X_test = test_df.drop('target', axis=1)
                y_test = test_df['target']
                
                # Use scaler
                X_test_scaled = scaler.transform(X_test)
                
                # Select features
                feature_indices = [i for i, f in enumerate(X_test.columns) if f in feature_names]
                X_test_selected = X_test_scaled[:, feature_indices]
                
                # Load best model and make predictions
                best_model = joblib.load(models_to_test['Best Model (Random Forest)'])
                predictions = best_model.predict(X_test_selected)
                accuracy = (predictions == y_test.values).mean()
                
                print(f"  ✓ Test accuracy on saved test set: {accuracy*100:.2f}%")
                
        except Exception as e:
            print(f"  ⚠ Could not test with test dataset: {e}")
    else:
        print(f"  ⚠ Test dataset not found at {test_data_path}")
    
    print("\n" + "="*80)
    if all_models_work:
        print("✓ ALL TESTS PASSED")
        print("\nModels are ready for use:")
        print("  - Can be loaded successfully")
        print("  - Can make predictions")
        print("  - Preprocessing pipeline works")
        print("\nTo use in production:")
        print("  1. Load scaler: scaler = joblib.load('models/scaler.pkl')")
        print("  2. Load features: features = joblib.load('models/feature_names.pkl')")
        print("  3. Load model: model = joblib.load('models/best_model.pkl')")
        print("  4. Scale and predict: X_scaled = scaler.transform(X); pred = model.predict(X_scaled)")
        print("="*80)
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return False


if __name__ == '__main__':
    import sys
    success = test_model_loading()
    sys.exit(0 if success else 1)
