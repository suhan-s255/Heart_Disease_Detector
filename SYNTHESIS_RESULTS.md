# Synthetic Data Generation Results

## Overview

This document summarizes the results of synthetic data generation and evaluation for the heart disease dataset.

## Generation Summary

- **Original dataset**: 1,888 rows
- **Target dataset**: 8,768 rows
- **Synthetic rows generated**: 10,320 rows (with 1.5x buffer for deduplication)
- **Duplicates removed**: 0
- **Final augmented dataset**: 8,768 rows ✅ (target achieved)
- **Output file**: `data/synthetic_augmented_heart_dataset.csv`

### Class Distribution

**Original dataset (cleaned_merged_heart_dataset.csv)**:
- Class 0 (No disease): 911 (48.25%)
- Class 1 (Disease): 977 (51.75%)

**Augmented dataset (synthetic_augmented_heart_dataset.csv)**:
- Class 0 (No disease): 3,410 (38.88%)
- Class 1 (Disease): 5,358 (61.12%)

Note: The class distribution shifted in the augmented dataset. This is a characteristic of CTGAN generation and may affect model performance.

## Quality Evaluation

### SDMetrics Evaluation

The SDV QualityReport API experienced compatibility issues with the installed version (SDMetrics 0.24.0). The fallback KS Complement metric computation also encountered issues, resulting in 0.0 scores for all columns.

**Note**: This is likely due to API changes in newer versions of SDMetrics. The metric computation itself may not be functioning correctly with the current SDV/SDMetrics versions.

### Machine Learning Performance Comparison

A Random Forest classifier was trained on both datasets with the following results:

#### Baseline (Cleaned Dataset - 1,888 rows)
- **Accuracy**: 0.9709 (97.09%)
- **Precision**: 0.9602 (96.02%)
- **Recall**: 0.9847 (98.47%)
- **F1-Score**: 0.9723 (97.23%)
- **AUC**: 0.9983 (99.83%)

#### Augmented (Synthetic Dataset - 8,768 rows)
- **Accuracy**: 0.7600 (76.00%)
- **Precision**: 0.7644 (76.44%)
- **Recall**: 0.8778 (87.78%)
- **F1-Score**: 0.8172 (81.72%)
- **AUC**: 0.8277 (82.77%)

#### Performance Difference
- **Accuracy**: -21.09% (exceeds ±3% threshold ⚠)
- **Precision**: -19.58% (exceeds ±3% threshold ⚠)
- **Recall**: -10.69% (exceeds ±3% threshold ⚠)
- **F1-Score**: -15.95% (exceeds ±3% threshold ⚠)
- **AUC**: -17.09% (exceeds ±3% threshold ⚠)

## Analysis and Findings

### Why is performance lower on the augmented dataset?

1. **Synthetic Data Quality**: CTGAN-generated synthetic data may not perfectly preserve all statistical relationships and patterns from the original data. This is a known limitation of generative models for tabular data.

2. **Class Imbalance Shift**: The original dataset was nearly balanced (48%/52%), but the augmented dataset shows a shift toward class 1 (39%/61%). This imbalance could affect model performance.

3. **Feature Relationships**: CTGAN may struggle to capture complex non-linear interactions between features, especially in medical datasets where relationships can be subtle.

4. **Dataset Size Comparison**: The baseline was trained on 1,888 high-quality real samples, while the augmented version includes 6,880 synthetic samples. The dilution of real data with lower-quality synthetic data affects overall performance.

## Recommendations

### For Production Use

1. **Use Original Data**: For critical applications like heart disease prediction, the original cleaned dataset (1,888 rows) provides better model performance and should be preferred.

2. **Augmentation Strategy**: If more data is needed, consider:
   - Collecting more real data
   - Using data augmentation techniques specific to tabular medical data
   - Applying SMOTE or similar oversampling techniques for class balancing
   - Using the synthetic data only for validation/testing, not training

3. **Hybrid Approach**: Train on the original dataset but use synthetic data for:
   - Stress testing model robustness
   - Generating edge cases
   - Privacy-preserving model sharing

### For Improving Synthetic Data Quality

1. **Increase Training Epochs**: The current CTGAN model uses 300 epochs. Increasing to 500-1000 epochs may improve quality (at the cost of longer training time).

2. **Conditional Generation**: Implement conditional CTGAN to generate samples with specific target class distributions matching the original data.

3. **Post-processing**: Apply statistical adjustments to ensure synthetic data matches the original distribution more closely.

4. **Try Alternative Methods**:
   - TVAE (Tabular VAE) from SDV
   - GaussianCopula synthesizer (faster, sometimes better for medical data)
   - CTAB-GAN or CTAB-GAN+ (improved versions of CTGAN)

## Acceptance Criteria Status

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Output row count | 8,768 | 8,768 | ✅ PASS |
| Column compatibility | Exact match | Exact match | ✅ PASS |
| Data type compatibility | Exact match | Exact match | ✅ PASS |
| SDV quality score | ≥ 0.70 | 0.00 (API issue) | ⚠ N/A |
| ML metrics within ±3% | ±3% | -10% to -21% | ❌ FAIL |

## Conclusion

The synthetic data generation pipeline successfully creates 8,768 rows with correct structure and data types. However, the quality of synthetic data is not sufficient to maintain ML model performance within the ±3% target.

**Recommendation**: Use the original cleaned dataset (1,888 rows) for production model training. The synthetic augmented dataset can be used for:
- Testing and development
- Privacy-preserving data sharing
- Model robustness evaluation
- Understanding CTGAN capabilities and limitations

The scripts and infrastructure provided allow for future improvements in synthetic data quality through parameter tuning and alternative synthesis methods.

## Files Generated

- `data/synthetic_augmented_heart_dataset.csv` - Augmented dataset with 8,768 rows
- `reports/synthetic_evaluation.json` - Machine-readable evaluation results
- `reports/synthetic_evaluation.txt` - Human-readable evaluation summary
- `scripts/generate_synthetic_compatible.py` - Generation script
- `scripts/evaluate_synthetic_and_model.py` - Evaluation script
- `tests/test_synthetic_generation.py` - Automated tests
- `data/SYNTHESIS_README.md` - Usage and installation guide
