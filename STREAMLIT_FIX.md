# Streamlit Feature Mismatch Fix

## Problem Statement

When running the Streamlit app with trained models, users encountered the following error:

```
ValueError: The feature names should match those that were passed during fit. 
Feature names must be in the same order as they were in fit.
```

## Root Cause Analysis

The issue had two main causes:

### 1. Polynomial Feature Name Mismatch

**Training Script (train_hrlfm_pipeline.py):**
- Uses `sklearn.preprocessing.PolynomialFeatures` to generate polynomial features
- Feature names are generated using `poly.get_feature_names_out()`
- Example: `'age^2'`, `'age trestbps'`, etc.

**Original Streamlit App:**
- Manually created polynomial feature names
- Used custom naming: `poly_{name1}^2`, `poly_{name1} {name2}`
- Names didn't match sklearn's format

### 2. Scaler-Feature Selection Mismatch

**Training Pipeline:**
1. Creates ALL features (base + engineered + polynomial) → ~50+ features
2. Fits scaler on ALL features
3. Applies feature selection to choose best ~36 features
4. Trains models on SELECTED features
5. Saves:
   - `scaler.pkl`: Expects ALL features
   - `feature_names.pkl`: Contains only SELECTED features

**Original Streamlit App:**
- Created all features
- Tried to scale using only SELECTED features
- Scaler rejected input because it was trained on ALL features

## Solution

### Fix 1: Use sklearn's PolynomialFeatures

**Before:**
```python
for name1, val1 in key_features.items():
    poly_features[f'poly_{name1}^2'] = val1 ** 2
    for name2, val2 in key_features.items():
        if name1 < name2:
            poly_features[f'poly_{name1} {name2}'] = val1 * val2
```

**After:**
```python
from sklearn.preprocessing import PolynomialFeatures

key_features_df = pd.DataFrame([key_features_dict])
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_data = poly.fit_transform(key_features_df)
poly_feature_names = poly.get_feature_names_out(key_features_df.columns)

for i, name in enumerate(poly_feature_names):
    if name not in key_features_df.columns:
        poly_features[f'poly_{name}'] = poly_data[0, i]
```

### Fix 2: Scale ALL Features, Then Select

**New Flow:**
1. Create ALL features (matching training)
2. Use `scaler.feature_names_in_` to get expected feature order
3. Reorder and provide ALL features to scaler
4. Scale the features
5. Select only the features the model expects
6. Pass to model for prediction

**Implementation:**
```python
if scaler is not None and hasattr(scaler, 'feature_names_in_'):
    scaler_features = list(scaler.feature_names_in_)
    
    # Ensure all scaler features are present
    for feat in scaler_features:
        if feat not in input_df.columns:
            input_df[feat] = 0
    
    # Scale ALL features
    input_df_for_scaling = input_df[scaler_features]
    input_scaled_all = scaler.transform(input_df_for_scaling)
    
    # Select only features the model was trained on
    if feature_names:
        selected_indices = [scaler_features.index(f) for f in feature_names if f in scaler_features]
        input_scaled = input_scaled_all[:, selected_indices]
```

## Testing

Unit tests have been added in `tests/test_feature_generation.py` to verify:
1. Polynomial feature names match between training and Streamlit
2. Base features are correctly generated
3. Engineered features are correctly generated

To run tests:
```bash
python -m unittest tests.test_feature_generation
```

## Verification Steps

To verify the fix works:

1. Train models (if not already done):
   ```bash
   python train_hrlfm_pipeline.py
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. Enter patient data and click "Predict Heart Disease Risk"

4. The prediction should complete without feature name mismatch errors

## Key Learnings

1. **Feature consistency is critical**: When using sklearn transformers like `PolynomialFeatures` or `RobustScaler`, the feature names and order must match exactly between training and inference.

2. **Understand the pipeline**: The training pipeline's feature selection happens AFTER scaling, so the scaler needs ALL features, not just selected ones.

3. **Use sklearn's feature_names_in_**: Modern sklearn (>=1.0) stores feature names, making it easier to debug and fix these issues.

4. **Test with actual data**: Unit tests help, but testing with the actual trained models is essential to catch these issues.

## Files Modified

- `app/streamlit_app.py`: Fixed polynomial feature generation and scaler handling
- `tests/test_feature_generation.py`: Added unit tests for feature generation consistency

## Compatibility

- Requires scikit-learn >= 1.0 (for `feature_names_in_` attribute)
- Compatible with all trained models from `train_hrlfm_pipeline.py`
- No changes needed to training script or saved models
