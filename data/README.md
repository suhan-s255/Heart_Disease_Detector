# Heart Disease Dataset - HRLFM Pipeline

## Overview
This directory contains the heart disease dataset used by the HRLFM (High-Resolution Logistic-Forest Model) pipeline:
- **cleaned_merged_heart_dataset.csv**: Cleaned and merged dataset with 1,888 patient records

## HRLFM Dataset: cleaned_merged_heart_dataset.csv

### Overview
This dataset contains 1,888 patient records with 13 clinical features used to predict the presence of heart disease. The dataset is cleaned, merged from multiple sources, and optimized for the HRLFM machine learning pipeline.

### Clinical Features

1. **age**: Age in years (29-77)
2. **sex**: Sex (0 = female, 1 = male)
3. **cp**: Chest pain type (0-4)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
   - 4: Other
4. **trestbps**: Resting blood pressure in mm Hg (94-200)
5. **chol**: Serum cholesterol in mg/dl (126-564)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
   - 0: Normal
   - 1: ST-T wave abnormality
   - 2: Left ventricular hypertrophy
8. **thalachh**: Maximum heart rate achieved (71-202)
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest (0-6.2)
11. **slope**: Slope of the peak exercise ST segment (0-3)
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
    - 3: Other
12. **ca**: Number of major vessels (0-4) colored by fluoroscopy
13. **thal**: Thalassemia (0-7)
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect
    - 3-7: Other categories

### Target Variable

**target**: Diagnosis of heart disease
- 0: No disease (< 50% diameter narrowing)
- 1: Disease present (> 50% diameter narrowing)

### Dataset Characteristics

- **Total samples**: 1,888
- **Training samples**: ~1,510 (80%)
- **Test samples**: ~378 (20%) - saved separately as test_dataset.csv
- **Original features**: 13
- **Engineered features**: 23 (created by HRLFM pipeline)
- **Final features used**: 36 (after feature selection)
- **Target classes**: 2 (Binary classification)
- **Class distribution**: Approximately 48% No disease, 52% Disease (balanced)
- **Missing values**: None

### Engineered Features (Created by HRLFM Pipeline)

#### Domain-Specific Features (8 features)
- **age_chol_interaction**: Age × Cholesterol
- **age_bp_interaction**: Age × Blood Pressure
- **bp_chol_ratio**: Blood Pressure / Cholesterol
- **heart_rate_reserve**: Max Heart Rate - Age
- **st_depression_severity**: ST Depression × (Slope + 1)
- **chest_pain_exercise_risk**: Chest Pain Type × (Exercise Angina + 1)
- **age_group**: Categorical age grouping (4 groups)
- **chol_category**: Categorical cholesterol grouping (3 groups)

#### Polynomial Features (15 features)
- Squared terms for: age, trestbps, chol, thalachh, oldpeak
- Interaction terms for all pairs of these 5 features

### Data Quality & Preprocessing

The HRLFM pipeline automatically handles:
- **Missing values**: Median imputation (numerical), mode imputation (categorical)
- **Outliers**: IQR-based capping with 3×IQR threshold
- **Class balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature scaling**: RobustScaler (less sensitive to outliers)
- **Feature selection**: SelectKBest + SelectFromModel (union approach)

### Pipeline Performance

Using this dataset, the HRLFM pipeline achieves:
- **Target**: ≥85% accuracy
- **Status**: ✅ Achieved across all models
- **Models**: 3 trained models (Logistic Regression, Random Forest, HRLFM)
- **Cross-validation**: 5-fold stratified CV for robust evaluation

### Top 15 Most Important Features

1. thal (10.2%)
2. chest_pain_exercise_risk (7.9%)
3. cp (7.1%)
4. slope (6.0%)
5. ca (4.2%)
6. thalachh (3.1%)
7. heart_rate_reserve (3.1%)
8. poly_age oldpeak (3.1%)
9. poly_trestbps chol (3.0%)
10. poly_oldpeak^2 (2.9%)
11. poly_thalachh^2 (2.7%)
12. poly_age thalachh (2.7%)
13. st_depression_severity (2.6%)
14. poly_trestbps oldpeak (2.5%)
15. poly_age trestbps (2.5%)

## Usage

To train models on this dataset:

```bash
# Using the automated script
bash run_hrlfm_pipeline.sh

# Or manually
python train_hrlfm_pipeline.py
```

For complete documentation, see [HRLFM_PIPELINE.md](../HRLFM_PIPELINE.md)

## Source

This dataset is a cleaned and merged version derived from heart disease datasets in the UCI Machine Learning Repository, optimized for the HRLFM pipeline.
