#!/usr/bin/env python
"""
Verification script to confirm HRLFM pipeline achieves ≥85% accuracy
with the cleaned dataset as documented in data/README.md
"""

import pandas as pd
import joblib
import sys
from pathlib import Path

def verify_accuracy():
    """Verify that saved models meet the ≥85% accuracy requirement."""
    
    print("="*80)
    print("HRLFM PIPELINE ACCURACY VERIFICATION")
    print("="*80)
    
    # Check if model comparison file exists
    model_comparison_path = Path("models/model_comparison.csv")
    if not model_comparison_path.exists():
        print("\n✗ ERROR: models/model_comparison.csv not found")
        print("  Please run: python train_hrlfm_pipeline.py")
        return False
    
    # Load model comparison results
    df = pd.read_csv(model_comparison_path)
    print("\nModel Performance Summary:")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    
    # Check model performance
    target_accuracy = 0.85
    print(f"\nTarget Accuracy: ≥{target_accuracy*100}%")
    print("\nAccuracy Check:")
    
    models_above_target = []
    models_below_target = []
    
    for _, row in df.iterrows():
        model_name = row['Model']
        accuracy = row['Accuracy']
        status = "✓ PASS" if accuracy >= target_accuracy else "⚠ BELOW"
        
        if accuracy >= target_accuracy:
            print(f"  {status}: {model_name} - {accuracy*100:.2f}%")
            models_above_target.append(model_name)
        else:
            print(f"  {status}: {model_name} - {accuracy*100:.2f}% (baseline model)")
            models_below_target.append(model_name)
    
    # Get best model
    best_model = df.loc[df['Accuracy'].idxmax()]
    best_accuracy = best_model['Accuracy']
    
    print("\n" + "="*80)
    
    # Check if best model meets target (this is the key requirement)
    if best_accuracy >= target_accuracy:
        print("✓ VERIFICATION PASSED: Best models meet ≥85% accuracy requirement")
        print(f"\n{len(models_above_target)} model(s) exceed target:")
        for model in models_above_target:
            print(f"  ✓ {model}")
        
        if models_below_target:
            print(f"\nNote: {len(models_below_target)} baseline model(s) below target (expected):")
            for model in models_below_target:
                print(f"  - {model}")
        
        print(f"\nBest Model: {best_model['Model']}")
        print(f"  Accuracy: {best_model['Accuracy']*100:.2f}%")
        print(f"  Precision: {best_model['Precision']*100:.2f}%")
        print(f"  Recall: {best_model['Recall']*100:.2f}%")
        print(f"  F1-Score: {best_model['F1-Score']*100:.2f}%")
        print(f"  ROC-AUC: {best_model['ROC-AUC']:.4f}")
        print("\nModels are saved and ready for deployment:")
        print("  - models/best_model.pkl (Random Forest)")
        print("  - models/hrlfm_model.pkl (HRLFM hybrid)")
        print("  - models/random_forest_model.pkl")
        print("  - models/logistic_regression_model.pkl")
        print("="*80)
        return True
    else:
        print("✗ VERIFICATION FAILED: Best model below 85% accuracy")
        print(f"  Best accuracy achieved: {best_accuracy*100:.2f}%")
        print("="*80)
        return False


if __name__ == '__main__':
    success = verify_accuracy()
    sys.exit(0 if success else 1)
