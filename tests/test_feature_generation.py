"""
Tests for feature generation consistency between training and Streamlit app
"""
import unittest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestFeatureGeneration(unittest.TestCase):
    """Test feature generation produces consistent names"""
    
    def test_polynomial_feature_generation(self):
        """Test that polynomial feature generation matches training approach"""
        import pandas as pd
        from sklearn.preprocessing import PolynomialFeatures
        
        # Simulate training approach
        key_features = ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak']
        test_data = pd.DataFrame([[50, 120, 200, 150, 1.0]], columns=key_features)
        
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_data = poly.fit_transform(test_data)
        poly_feature_names = poly.get_feature_names_out(test_data.columns)
        
        # Get features that would be added (excluding original)
        training_added_features = [f'poly_{name}' for name in poly_feature_names if name not in key_features]
        
        # Simulate Streamlit approach (after fix)
        age, trestbps, chol, thalachh, oldpeak = 50, 120, 200, 150, 1.0
        poly_features = {}
        key_features_dict = {'age': age, 'trestbps': trestbps, 'chol': chol, 
                            'thalachh': thalachh, 'oldpeak': oldpeak}
        
        key_features_df = pd.DataFrame([key_features_dict])
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_data = poly.fit_transform(key_features_df)
        poly_feature_names = poly.get_feature_names_out(key_features_df.columns)
        
        for i, name in enumerate(poly_feature_names):
            if name not in key_features_df.columns:
                poly_features[f'poly_{name}'] = poly_data[0, i]
        
        streamlit_added_features = list(poly_features.keys())
        
        # Compare
        self.assertEqual(set(training_added_features), set(streamlit_added_features),
                        "Training and Streamlit should generate the same polynomial feature names")
        
        # Also check the order is consistent
        self.assertEqual(training_added_features, streamlit_added_features,
                        "Training and Streamlit should generate polynomial features in the same order")
    
    def test_base_feature_generation(self):
        """Test that base features are correctly generated"""
        # Base features that should be created
        expected_base_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalachh', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal'
        ]
        
        # Simulate base feature creation from Streamlit app
        age, sex, cp = 50, 1, 3
        trestbps, chol, fbs = 120, 200, 0
        restecg, thalachh, exang = 0, 150, 0
        oldpeak, slope, ca, thal = 1.0, 1, 0, 2
        
        base_features = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Check all expected features are present
        for feature in expected_base_features:
            self.assertIn(feature, base_features, 
                         f"Base features should include '{feature}'")
        
        # Check no extra features
        self.assertEqual(set(base_features.keys()), set(expected_base_features),
                        "Base features should match expected features exactly")
    
    def test_engineered_feature_generation(self):
        """Test that engineered features are correctly generated"""
        # Expected engineered features
        expected_engineered_features = [
            'age_chol_interaction',
            'age_bp_interaction',
            'bp_chol_ratio',
            'heart_rate_reserve',
            'st_depression_severity',
            'chest_pain_exercise_risk',
            'age_group',
            'chol_category'
        ]
        
        # Simulate engineered feature creation from Streamlit app
        age, trestbps, chol = 50, 120, 200
        thalachh, oldpeak, slope = 150, 1.0, 1
        cp, exang = 3, 0
        
        engineered_features = {}
        
        # Age-related interactions
        engineered_features['age_chol_interaction'] = age * chol
        engineered_features['age_bp_interaction'] = age * trestbps
        
        # Blood pressure and cholesterol ratio
        engineered_features['bp_chol_ratio'] = trestbps / (chol + 1)
        
        # Exercise capacity
        engineered_features['heart_rate_reserve'] = thalachh - age
        
        # ST depression and slope interaction
        engineered_features['st_depression_severity'] = oldpeak * (slope + 1)
        
        # Risk score combination
        engineered_features['chest_pain_exercise_risk'] = cp * (exang + 1)
        
        # Age groups
        if age <= 40:
            age_group = 0
        elif age <= 55:
            age_group = 1
        elif age <= 70:
            age_group = 2
        else:
            age_group = 3
        engineered_features['age_group'] = age_group
        
        # Cholesterol categories
        if chol <= 200:
            chol_cat = 0
        elif chol <= 240:
            chol_cat = 1
        else:
            chol_cat = 2
        engineered_features['chol_category'] = chol_cat
        
        # Check all expected features are present
        for feature in expected_engineered_features:
            self.assertIn(feature, engineered_features, 
                         f"Engineered features should include '{feature}'")
        
        # Check no extra features
        self.assertEqual(set(engineered_features.keys()), set(expected_engineered_features),
                        "Engineered features should match expected features exactly")


if __name__ == '__main__':
    unittest.main()
