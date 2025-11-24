#!/usr/bin/env python3
"""
Synthetic Data Evaluation Script for Heart Disease Dataset

This script evaluates the quality of synthetic data and compares ML model
performance between the original cleaned dataset and the synthetic augmented dataset.

Requirements:
    - Run generate_synthetic_compatible.py first to create synthetic data
    - Install synthesis requirements: pip install -r requirements-synthesis.txt

Usage:
    python scripts/evaluate_synthetic_and_model.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
CLEANED_DATA_PATH = 'data/cleaned_merged_heart_dataset.csv'
SYNTHETIC_DATA_PATH = 'data/synthetic_augmented_heart_dataset.csv'
REPORTS_DIR = 'reports'
JSON_REPORT_PATH = os.path.join(REPORTS_DIR, 'synthetic_evaluation.json')
TEXT_REPORT_PATH = os.path.join(REPORTS_DIR, 'synthetic_evaluation.txt')
RANDOM_SEED = 42

# Set random seeds
np.random.seed(RANDOM_SEED)


def check_dependencies():
    """Check if SDMetrics is available for quality evaluation."""
    try:
        import sdmetrics
        print(f"✓ SDMetrics {sdmetrics.__version__} found")
        return True
    except ImportError:
        print("✗ SDMetrics not found - quality evaluation will be skipped")
        print("  Install with: pip install sdmetrics")
        return False


def load_datasets():
    """Load both cleaned and synthetic datasets."""
    print("Loading datasets...")
    
    if not os.path.exists(CLEANED_DATA_PATH):
        print(f"✗ Error: Cleaned dataset not found: {CLEANED_DATA_PATH}")
        sys.exit(1)
    
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        print(f"✗ Error: Synthetic dataset not found: {SYNTHETIC_DATA_PATH}")
        print("  Run generate_synthetic_compatible.py first!")
        sys.exit(1)
    
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    synthetic_df = pd.read_csv(SYNTHETIC_DATA_PATH)
    
    print(f"✓ Cleaned dataset: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
    print(f"✓ Synthetic dataset: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns")
    
    return cleaned_df, synthetic_df


def detect_target_column(df):
    """Detect the target column in the dataset."""
    target_candidates = ['target', 'HeartDisease', 'heart_disease', 'label']
    for col in target_candidates:
        if col in df.columns:
            return col
    
    print("✗ Error: No target column found!")
    sys.exit(1)


def evaluate_data_quality(cleaned_df, synthetic_df):
    """Evaluate synthetic data quality using SDMetrics."""
    print("\n" + "="*60)
    print("DATA QUALITY EVALUATION")
    print("="*60)
    
    try:
        from sdmetrics.reports.single_table import QualityReport
        from sdmetrics.single_table import NewRowSynthesis, KSComplement
        
        print("\nComputing SDV Quality Metrics...")
        
        # Use individual metrics if QualityReport fails
        try:
            # Try newer API with metadata
            from sdv.metadata import SingleTableMetadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(cleaned_df)
            
            report = QualityReport()
            report.generate(cleaned_df, synthetic_df, metadata.to_dict(), verbose=False)
            
            overall_score = report.get_score()
            properties = report.get_properties()
            details = report.get_details()
            
            print(f"\n✓ Overall Quality Score: {overall_score:.4f}")
            print("\nProperty Scores:")
            for prop, score in properties.items():
                print(f"  {prop}: {score:.4f}")
            
            quality_results = {
                'overall_score': float(overall_score),
                'properties': {k: float(v) for k, v in properties.items()},
                'details': details.to_dict()
            }
            
        except Exception as e1:
            print(f"  QualityReport API failed, using individual metrics: {e1}")
            
            # Fallback: use individual metrics
            column_scores = {}
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    try:
                        score = KSComplement.compute(
                            real_data=cleaned_df[col],
                            synthetic_data=synthetic_df[col]
                        )
                        column_scores[col] = float(score)
                    except:
                        column_scores[col] = 0.0
            
            # Calculate average score
            overall_score = sum(column_scores.values()) / len(column_scores) if column_scores else 0.0
            
            print(f"\n✓ Average KS Complement Score: {overall_score:.4f}")
            print("\nColumn Scores (KS Complement):")
            for col, score in column_scores.items():
                print(f"  {col}: {score:.4f}")
            
            quality_results = {
                'overall_score': float(overall_score),
                'column_scores': column_scores,
                'metric': 'KS Complement (higher is better)'
            }
        
        return quality_results
        
    except Exception as e:
        print(f"✗ Quality evaluation failed: {e}")
        print("  Continuing without quality metrics...")
        return None


def train_and_evaluate_model(X, y, dataset_name):
    """Train a Random Forest model and return evaluation metrics."""
    print(f"\nTraining Random Forest on {dataset_name}...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with fixed random_state for reproducibility
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n  Results for {dataset_name}:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print(f"    AUC:       {metrics['auc']:.4f}")
    
    return metrics, rf_model


def compare_model_performance(baseline_metrics, augmented_metrics):
    """Compare metrics between baseline and augmented datasets."""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    comparison = {}
    
    for metric_name in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric_name]
        augmented_val = augmented_metrics[metric_name]
        diff = augmented_val - baseline_val
        pct_diff = (diff / baseline_val) * 100 if baseline_val != 0 else 0
        
        comparison[metric_name] = {
            'baseline': baseline_val,
            'augmented': augmented_val,
            'difference': diff,
            'percent_difference': pct_diff
        }
        
        status = "✓" if abs(diff) <= 0.03 else "⚠"
        print(f"\n{metric_name.upper()}:")
        print(f"  Baseline:   {baseline_val:.4f}")
        print(f"  Augmented:  {augmented_val:.4f}")
        print(f"  Difference: {diff:+.4f} ({pct_diff:+.2f}%) {status}")
    
    # Overall assessment
    print("\n" + "="*60)
    max_diff = max(abs(comp['difference']) for comp in comparison.values())
    
    if max_diff <= 0.03:
        print("✓ PASS: All metrics within ±3% absolute difference")
    else:
        print("⚠ WARNING: Some metrics exceed ±3% absolute difference")
    
    return comparison


def save_results(quality_results, baseline_metrics, augmented_metrics, comparison):
    """Save evaluation results to JSON and text files."""
    print(f"\nSaving results to {REPORTS_DIR}/...")
    
    # Create reports directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'quality_evaluation': quality_results,
        'ml_evaluation': {
            'baseline': baseline_metrics,
            'augmented': augmented_metrics,
            'comparison': comparison
        }
    }
    
    # Save JSON
    with open(JSON_REPORT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON report: {JSON_REPORT_PATH}")
    
    # Save text report
    with open(TEXT_REPORT_PATH, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SYNTHETIC DATA EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Quality evaluation
        if quality_results:
            f.write("DATA QUALITY EVALUATION\n")
            f.write("-"*60 + "\n")
            f.write(f"Overall Quality Score: {quality_results['overall_score']:.4f}\n\n")
            
            if 'properties' in quality_results:
                f.write("Property Scores:\n")
                for prop, score in quality_results['properties'].items():
                    f.write(f"  {prop}: {score:.4f}\n")
            elif 'column_scores' in quality_results:
                f.write(f"Metric: {quality_results.get('metric', 'Unknown')}\n")
                f.write("Column Scores:\n")
                for col, score in quality_results['column_scores'].items():
                    f.write(f"  {col}: {score:.4f}\n")
            f.write("\n")
        
        # ML evaluation
        f.write("MACHINE LEARNING EVALUATION\n")
        f.write("-"*60 + "\n\n")
        
        f.write("Baseline (Cleaned Dataset):\n")
        for metric, value in baseline_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Augmented (Synthetic Dataset):\n")
        for metric, value in augmented_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Comparison:\n")
        for metric, values in comparison.items():
            f.write(f"  {metric}:\n")
            f.write(f"    Difference: {values['difference']:+.4f}\n")
            f.write(f"    Percent Difference: {values['percent_difference']:+.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        
        max_diff = max(abs(comp['difference']) for comp in comparison.values())
        if max_diff <= 0.03:
            f.write("RESULT: PASS - All metrics within ±3% absolute difference\n")
        else:
            f.write("RESULT: WARNING - Some metrics exceed ±3% absolute difference\n")
        
        f.write("="*60 + "\n")
    
    print(f"✓ Saved text report: {TEXT_REPORT_PATH}")


def main():
    """Main execution function."""
    print("="*60)
    print("SYNTHETIC DATA EVALUATION")
    print("="*60)
    
    # Check dependencies
    has_sdmetrics = check_dependencies()
    
    # Load datasets
    cleaned_df, synthetic_df = load_datasets()
    
    # Detect target column
    target_col = detect_target_column(cleaned_df)
    print(f"\n✓ Target column: '{target_col}'")
    
    # Evaluate data quality if SDMetrics is available
    quality_results = None
    if has_sdmetrics:
        quality_results = evaluate_data_quality(cleaned_df, synthetic_df)
    else:
        print("\nSkipping quality evaluation (SDMetrics not available)")
    
    # Prepare data for ML evaluation
    print("\n" + "="*60)
    print("MACHINE LEARNING EVALUATION")
    print("="*60)
    
    # Baseline: Cleaned dataset
    X_baseline = cleaned_df.drop(columns=[target_col])
    y_baseline = cleaned_df[target_col]
    baseline_metrics, _ = train_and_evaluate_model(
        X_baseline, y_baseline, "Cleaned Dataset (Baseline)"
    )
    
    # Prepare augmented data with sample weights (prioritize real data)
    print("\n" + "-"*60)
    print("AUGMENTED: Train with sample weighting (real data prioritized)")
    print("-"*60)
    
    # Split original data for fair comparison
    X_train_clean, X_test_common, y_train_clean, y_test_common = train_test_split(
        X_baseline, y_baseline, test_size=0.2, random_state=RANDOM_SEED, stratify=y_baseline
    )
    
    # Get synthetic-only rows (exclude original 1888 rows)
    synthetic_only = synthetic_df.iloc[len(cleaned_df):]
    X_synthetic = synthetic_only.drop(columns=[target_col])
    y_synthetic = synthetic_only[target_col]
    
    # Combine training data
    X_train_combined = pd.concat([X_train_clean, X_synthetic], ignore_index=True)
    y_train_combined = pd.concat([y_train_clean, y_synthetic], ignore_index=True)
    
    # Create sample weights: real data gets 3x weight vs synthetic
    sample_weights = np.ones(len(y_train_combined))
    sample_weights[:len(y_train_clean)] = 3.0  # Real data weight
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_common)
    
    # Train with sample weights
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train_combined, sample_weight=sample_weights)
    
    # Evaluate on common test set
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    augmented_metrics = {
        'accuracy': accuracy_score(y_test_common, y_pred),
        'precision': precision_score(y_test_common, y_pred, average='binary'),
        'recall': recall_score(y_test_common, y_pred, average='binary'),
        'f1': f1_score(y_test_common, y_pred, average='binary'),
        'auc': roc_auc_score(y_test_common, y_pred_proba)
    }
    
    print(f"  Training samples: {len(X_train_combined)} ({len(X_train_clean)} real [3x weight] + {len(X_synthetic)} synthetic [1x weight])")
    print(f"  Test samples: {len(X_test_common)} (same as baseline)")
    print(f"\n  Results:")
    print(f"    Accuracy:  {augmented_metrics['accuracy']:.4f}")
    print(f"    Precision: {augmented_metrics['precision']:.4f}")
    print(f"    Recall:    {augmented_metrics['recall']:.4f}")
    print(f"    F1-Score:  {augmented_metrics['f1']:.4f}")
    print(f"    AUC:       {augmented_metrics['auc']:.4f}")
    
    # Compare performance
    comparison = compare_model_performance(baseline_metrics, augmented_metrics)
    
    # Save results
    save_results(quality_results, baseline_metrics, augmented_metrics, comparison)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Reports saved to: {REPORTS_DIR}/")
    print(f"  - {JSON_REPORT_PATH}")
    print(f"  - {TEXT_REPORT_PATH}")


if __name__ == "__main__":
    main()
