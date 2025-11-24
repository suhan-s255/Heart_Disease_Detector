import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #E74C3C;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #D5F4E6;
        border: 2px solid #27AE60;
    }
    .at-risk {
        background-color: #FADBD8;
        border: 2px solid #E74C3C;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-based Heart Disease Detection System</p>', unsafe_allow_html=True)

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    """Load pre-trained models and preprocessing objects"""
    model_path = Path(__file__).parent.parent / 'models'
    
    models = {}
    try:
        if (model_path / 'logistic_regression_model.pkl').exists():
            models['Logistic Regression'] = joblib.load(model_path / 'logistic_regression_model.pkl')
        if (model_path / 'random_forest_model.pkl').exists():
            models['Random Forest'] = joblib.load(model_path / 'random_forest_model.pkl')
        if (model_path / 'hrlfm_model.pkl').exists():
            models['HRLFM'] = joblib.load(model_path / 'hrlfm_model.pkl')
        if (model_path / 'best_model.pkl').exists():
            models['Best Model'] = joblib.load(model_path / 'best_model.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None
    
    # Load preprocessing objects
    scaler = None
    feature_names = None
    
    try:
        if (model_path / 'scaler.pkl').exists():
            scaler = joblib.load(model_path / 'scaler.pkl')
        if (model_path / 'feature_names.pkl').exists():
            feature_names = joblib.load(model_path / 'feature_names.pkl')
    except Exception as e:
        st.warning(f"Preprocessing objects not fully loaded: {e}")
    
    return models, scaler, feature_names

# Sidebar - Model Selection
st.sidebar.title("⚙️ Configuration")
models, scaler, feature_names = load_models()

if models:
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        index=0
    )
    model = models[selected_model_name]
    
    # Sidebar - Information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info(
        "This application predicts the likelihood of heart disease based on various clinical parameters. "
        "The models were trained on the cleaned merged heart disease dataset with 1,888 patient records using "
        "advanced ML techniques including HRLFM (High-Resolution Logistic-Forest Model)."
    )
    
    st.sidebar.markdown("### 🎯 Model Info")
    st.sidebar.success(f"Currently using: **{selected_model_name}**")
else:
    st.error("No models found. Please run the training script first: `python train_hrlfm_pipeline.py`")
    st.stop()

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Model Performance", "ℹ️ About Dataset"])

with tab1:
    st.markdown("### Enter Patient Information")
    st.markdown("*Please enter clinical measurements from patient examination*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1, 
                             help="Patient's age in years")
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female",
                          help="Biological sex: 1 = Male, 0 = Female")
        
        st.markdown("**Chest Pain Type**")
        cp = st.selectbox("Chest Pain Type (cp)", 
                         options=[0, 1, 2, 3, 4],
                         format_func=lambda x: {
                             0: "0 - Typical Angina",
                             1: "1 - Atypical Angina", 
                             2: "2 - Non-anginal Pain",
                             3: "3 - Asymptomatic",
                             4: "4 - Other"
                         }.get(x, str(x)),
                         help="Type of chest pain experienced")
        
        st.markdown("**Cardiovascular Metrics**")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=80, max_value=200, value=120, step=1,
                                   help="Resting blood pressure in mm Hg")
        chol = st.number_input("Serum Cholesterol (mg/dl)", 
                              min_value=100, max_value=600, value=200, step=1,
                              help="Serum cholesterol in mg/dl")
    
    with col2:
        st.markdown("**Blood Sugar & ECG**")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                          options=[0, 1],
                          format_func=lambda x: "Yes" if x == 1 else "No",
                          help="1 if fasting blood sugar > 120 mg/dl, 0 otherwise")
        restecg = st.selectbox("Resting ECG Results", 
                              options=[0, 1, 2],
                              format_func=lambda x: {
                                  0: "0 - Normal",
                                  1: "1 - ST-T Wave Abnormality",
                                  2: "2 - Left Ventricular Hypertrophy"
                              }.get(x, str(x)),
                              help="Resting electrocardiographic results")
        
        st.markdown("**Exercise Metrics**")
        thalachh = st.number_input("Maximum Heart Rate Achieved", 
                                   min_value=60, max_value=220, value=150, step=1,
                                   help="Maximum heart rate achieved during exercise")
        exang = st.selectbox("Exercise Induced Angina", 
                            options=[0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No",
                            help="Exercise induced angina: 1 = Yes, 0 = No")
        oldpeak = st.number_input("ST Depression (oldpeak)", 
                                 min_value=0.0, max_value=7.0, value=1.0, step=0.1,
                                 help="ST depression induced by exercise relative to rest")
    
    with col3:
        st.markdown("**Additional Diagnostics**")
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                            options=[0, 1, 2, 3],
                            format_func=lambda x: {
                                0: "0 - Upsloping",
                                1: "1 - Flat",
                                2: "2 - Downsloping",
                                3: "3 - Unknown"
                            }.get(x, str(x)),
                            help="Slope of the peak exercise ST segment")
        ca = st.selectbox("Number of Major Vessels (0-4)", 
                         options=[0, 1, 2, 3, 4],
                         help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox("Thalassemia", 
                           options=[0, 1, 2, 3, 7],
                           format_func=lambda x: {
                               0: "0 - Unknown",
                               1: "1 - Normal",
                               2: "2 - Fixed Defect",
                               3: "3 - Reversible Defect",
                               7: "7 - Other"
                           }.get(x, str(x)),
                           help="Thalassemia: blood disorder")
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
        # Prepare base features
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
        
        # Create engineered features (same as in training pipeline)
        engineered_features = {}
        
        # Age-related interactions
        engineered_features['age_chol_interaction'] = age * chol
        engineered_features['age_bp_interaction'] = age * trestbps
        
        # Blood pressure and cholesterol ratio
        engineered_features['bp_chol_ratio'] = trestbps / (chol + 1)
        
        # Exercise capacity (max heart rate - age)
        engineered_features['heart_rate_reserve'] = thalachh - age
        
        # ST depression and slope interaction
        engineered_features['st_depression_severity'] = oldpeak * (slope + 1)
        
        # Risk score combination
        engineered_features['chest_pain_exercise_risk'] = cp * (exang + 1)
        
        # Age groups (encoded as in training)
        if age <= 40:
            age_group = 0  # young
        elif age <= 55:
            age_group = 1  # middle
        elif age <= 70:
            age_group = 2  # senior
        else:
            age_group = 3  # elderly
        engineered_features['age_group'] = age_group
        
        # Cholesterol categories (encoded as in training)
        if chol <= 200:
            chol_cat = 0  # normal
        elif chol <= 240:
            chol_cat = 1  # borderline
        else:
            chol_cat = 2  # high
        engineered_features['chol_category'] = chol_cat
        
        # Polynomial features (degree 2) - using PolynomialFeatures to match training
        poly_features = {}
        key_features_dict = {'age': age, 'trestbps': trestbps, 'chol': chol, 'thalachh': thalachh, 'oldpeak': oldpeak}
        
        # Create a small dataframe for polynomial feature generation
        key_features_df = pd.DataFrame([key_features_dict])
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_data = poly.fit_transform(key_features_df)
        poly_feature_names = poly.get_feature_names_out(key_features_df.columns)
        
        # Add polynomial features (excluding original features)
        for i, name in enumerate(poly_feature_names):
            if name not in key_features_df.columns:
                poly_features[f'poly_{name}'] = poly_data[0, i]
        
        # Combine all features
        all_features = {**base_features, **engineered_features, **poly_features}
        
        # Create DataFrame with all features
        input_df = pd.DataFrame([all_features])
        
        # Scale features
        try:
            if scaler is not None:
                # The scaler was fit on all features, so we need to ensure we have them in the right order
                # Get the feature names the scaler expects (from when it was fit)
                if hasattr(scaler, 'feature_names_in_'):
                    scaler_features = list(scaler.feature_names_in_)
                    # Make sure we have all required features
                    for feat in scaler_features:
                        if feat not in input_df.columns:
                            input_df[feat] = 0  # Default value for missing features
                    # Reorder columns to match scaler's expected order and transform
                    input_df_for_scaling = input_df[scaler_features]
                    input_scaled_all = scaler.transform(input_df_for_scaling)
                    
                    # Now select only the features that the model was trained on
                    if feature_names:
                        selected_indices = [scaler_features.index(f) for f in feature_names if f in scaler_features]
                        input_scaled = input_scaled_all[:, selected_indices]
                    else:
                        input_scaled = input_scaled_all
                else:
                    # Fallback: scaler doesn't have feature_names_in_ (shouldn't happen with sklearn >= 1.0)
                    # Assume feature_names has the right features
                    if feature_names:
                        for feat in feature_names:
                            if feat not in input_df.columns:
                                input_df[feat] = 0
                        input_df = input_df[feature_names]
                    input_scaled = scaler.transform(input_df)
            else:
                # If no scaler, just use the selected features
                if feature_names:
                    for feat in feature_names:
                        if feat not in input_df.columns:
                            input_df[feat] = 0
                    input_df = input_df[feature_names]
                input_scaled = input_df.values
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 0:
                    st.markdown(
                        '<div class="prediction-box healthy">'
                        '<h2 style="color: #27AE60; text-align: center;">✅ Low Risk</h2>'
                        '<p style="text-align: center; font-size: 1.2rem;">No significant heart disease risk detected</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box at-risk">'
                        '<h2 style="color: #E74C3C; text-align: center;">⚠️ High Risk</h2>'
                        '<p style="text-align: center; font-size: 1.2rem;">Heart disease risk detected - Please consult a healthcare professional</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown("#### Probability Breakdown")
                st.metric("No Disease Probability", f"{probability[0]*100:.2f}%")
                st.metric("Disease Probability", f"{probability[1]*100:.2f}%")
                
                # Probability bar chart
                prob_df = pd.DataFrame({
                    'Category': ['No Disease', 'Disease'],
                    'Probability': [probability[0]*100, probability[1]*100]
                })
                st.bar_chart(prob_df.set_index('Category'))
            
            # Show top contributing features
            st.markdown("---")
            st.markdown("#### 📊 Input Summary")
            summary_data = {
                'Parameter': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                             'Max Heart Rate', 'Exercise Angina', 'ST Depression'],
                'Value': [
                    f"{age} years",
                    "Male" if sex == 1 else "Female",
                    cp,
                    f"{trestbps} mm Hg",
                    f"{chol} mg/dl",
                    f"{thalachh} bpm",
                    "Yes" if exang == 1 else "No",
                    f"{oldpeak}"
                ]
            }
            st.table(pd.DataFrame(summary_data))
            
            # Disclaimer
            st.markdown("---")
            st.warning("⚠️ **Disclaimer**: This is a prediction tool and should not replace professional medical advice. "
                      "Please consult with a healthcare professional for proper diagnosis and treatment.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please ensure all required fields are filled correctly.")
            import traceback
            st.error(traceback.format_exc())

with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    
    # Try to load model comparison
    try:
        model_path = Path(__file__).parent.parent / 'models'
        if (model_path / 'model_comparison.csv').exists():
            comparison_df = pd.read_csv(model_path / 'model_comparison.csv')
            st.markdown("#### Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize metrics
            st.markdown("#### Performance Metrics Visualization")
            
            # Create bar charts for each metric
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Accuracy' in comparison_df.columns:
                    st.markdown("**Accuracy by Model**")
                    chart_data = comparison_df.set_index('Model')[['Accuracy']]
                    st.bar_chart(chart_data)
                
                if 'Precision' in comparison_df.columns:
                    st.markdown("**Precision by Model**")
                    chart_data = comparison_df.set_index('Model')[['Precision']]
                    st.bar_chart(chart_data)
                
                if 'F1-Score' in comparison_df.columns:
                    st.markdown("**F1-Score by Model**")
                    chart_data = comparison_df.set_index('Model')[['F1-Score']]
                    st.bar_chart(chart_data)
            
            with col2:
                if 'Recall' in comparison_df.columns:
                    st.markdown("**Recall by Model**")
                    chart_data = comparison_df.set_index('Model')[['Recall']]
                    st.bar_chart(chart_data)
                
                if 'ROC-AUC' in comparison_df.columns:
                    st.markdown("**ROC-AUC by Model**")
                    chart_data = comparison_df.set_index('Model')[['ROC-AUC']]
                    st.bar_chart(chart_data)
            
            # Show visualizations if available
            st.markdown("---")
            st.markdown("#### Additional Visualizations")
            
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                if (model_path / 'confusion_matrix.png').exists():
                    st.markdown("**Confusion Matrix**")
                    st.image(str(model_path / 'confusion_matrix.png'), use_column_width=True)
            
            with viz_col2:
                if (model_path / 'roc_curve.png').exists():
                    st.markdown("**ROC Curve**")
                    st.image(str(model_path / 'roc_curve.png'), use_column_width=True)
            
            with viz_col3:
                if (model_path / 'feature_importance.png').exists():
                    st.markdown("**Feature Importance**")
                    st.image(str(model_path / 'feature_importance.png'), use_column_width=True)
        else:
            st.info("Model comparison data not available. Run the training script to generate it.")
    except Exception as e:
        st.warning(f"Could not load model comparison: {e}")
    
    st.markdown("""
    ---
    ### 📚 Evaluation Metrics Explained
    
    Understanding the performance metrics used to evaluate our heart disease prediction models:
    
    #### **Accuracy**
    The proportion of correct predictions (both positive and negative) among the total number of cases examined.
    
    **Formula:**
    ```
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    ```
    
    Where:
    - TP = True Positives (correctly predicted disease cases)
    - TN = True Negatives (correctly predicted non-disease cases)
    - FP = False Positives (incorrectly predicted disease cases)
    - FN = False Negatives (incorrectly predicted non-disease cases)
    
    **Interpretation:** Higher is better. 90% accuracy means the model correctly predicts 90 out of 100 cases.
    
    ---
    
    #### **Precision** (Positive Predictive Value)
    The proportion of positive predictions that were actually correct. It answers: "Of all the patients we predicted have heart disease, how many actually have it?"
    
    **Formula:**
    ```
    Precision = TP / (TP + FP)
    ```
    
    **Interpretation:** Higher is better. High precision means fewer false alarms. Important when the cost of false positives is high.
    
    ---
    
    #### **Recall** (Sensitivity or True Positive Rate)
    The proportion of actual positive cases that were correctly identified. It answers: "Of all the patients who actually have heart disease, how many did we correctly identify?"
    
    **Formula:**
    ```
    Recall = TP / (TP + FN)
    ```
    
    **Interpretation:** Higher is better. High recall means we catch most disease cases. Critical in medical diagnosis where missing a disease case (false negative) can be dangerous.
    
    ---
    
    #### **F1-Score**
    The harmonic mean of precision and recall, providing a single score that balances both metrics.
    
    **Formula:**
    ```
    F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
    ```
    
    Or equivalently:
    ```
    F1-Score = 2TP / (2TP + FP + FN)
    ```
    
    **Interpretation:** Higher is better. F1-Score is useful when you need to balance precision and recall. It reaches its best value at 1 (perfect precision and recall) and worst at 0.
    
    ---
    
    #### **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve)
    Measures the model's ability to distinguish between classes across all possible classification thresholds.
    
    **ROC Curve:** Plots True Positive Rate (Recall) vs False Positive Rate at various threshold settings.
    
    **Formula for False Positive Rate:**
    ```
    FPR = FP / (FP + TN)
    ```
    
    **AUC Interpretation:**
    - AUC = 1.0: Perfect classifier
    - AUC = 0.9-1.0: Excellent
    - AUC = 0.8-0.9: Good
    - AUC = 0.7-0.8: Fair
    - AUC = 0.5: No better than random guessing
    
    The ROC-AUC score represents the probability that the model ranks a random positive example higher than a random negative example.
    
    ---
    
    ### 🎯 Choosing the Right Metric
    
    - **Accuracy**: Use when classes are balanced and all types of errors are equally important
    - **Precision**: Use when false positives are costly (e.g., unnecessary treatments)
    - **Recall**: Use when false negatives are costly (e.g., missing a disease diagnosis)
    - **F1-Score**: Use when you need a balance between precision and recall
    - **ROC-AUC**: Use to evaluate overall model performance across all thresholds; best for comparing models
    
    ### Models Overview:
    
    1. **Logistic Regression**: Fast, interpretable linear model suitable for understanding feature relationships. 
       Provides baseline performance and helps identify important risk factors.
    
    2. **Random Forest**: Robust ensemble method combining multiple decision trees. Excellent performance 
       through averaging predictions and reducing overfitting. Strong at capturing non-linear relationships.
    
    3. **HRLFM (High-Resolution Logistic-Forest Model)**: Hybrid approach combining Logistic Regression 
       and Random Forest through ensemble voting. Balances linear interpretability with non-linear 
       predictive power for robust, explainable predictions.
    
    ### Training Details:
    
    - **Dataset**: 1,888 patient records from cleaned merged heart disease dataset
    - **Features**: 13 original + 23 engineered = 36 selected features
    - **Cross-Validation**: 5-fold stratified cross-validation
    - **Hyperparameter Tuning**: RandomizedSearchCV
    - **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
    - **Feature Scaling**: RobustScaler (less sensitive to outliers)
    - **Target Achievement**: ✅ Exceeded 85% accuracy target
    """)

with tab3:
    st.markdown("### 📋 Dataset Information")
    
    st.markdown("""
    #### Features Description:
    
    This model was trained on the **cleaned merged heart disease dataset** containing 1,888 patient records 
    with 14 clinical features.
    
    **Clinical Features:**
    
    1. **age**: Age in years
    2. **sex**: Sex (1 = male, 0 = female)
    3. **cp**: Chest pain type
       - 0: Typical angina
       - 1: Atypical angina
       - 2: Non-anginal pain
       - 3: Asymptomatic
    4. **trestbps**: Resting blood pressure (mm Hg)
    5. **chol**: Serum cholesterol (mg/dl)
    6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    7. **restecg**: Resting electrocardiographic results
       - 0: Normal
       - 1: ST-T wave abnormality
       - 2: Left ventricular hypertrophy
    8. **thalachh**: Maximum heart rate achieved
    9. **exang**: Exercise induced angina (1 = yes, 0 = no)
    10. **oldpeak**: ST depression induced by exercise relative to rest
    11. **slope**: Slope of the peak exercise ST segment
        - 0: Upsloping
        - 1: Flat
        - 2: Downsloping
    12. **ca**: Number of major vessels (0-4) colored by fluoroscopy
    13. **thal**: Thalassemia
        - 0: Unknown
        - 1: Normal
        - 2: Fixed defect
        - 3: Reversible defect
    
    **Engineered Features:**
    The model uses 36 carefully engineered features including:
    - Age-related interactions (age × cholesterol, age × blood pressure)
    - Cardiovascular ratios (BP/cholesterol, heart rate reserve)
    - ST depression severity indicators
    - Chest pain-exercise risk scores
    - Age and cholesterol categories
    - Polynomial features (squared terms and interactions)
    
    #### Target Variable:
    - **target**: Heart disease diagnosis (0 = No disease, 1 = Disease)
    
    #### Dataset Statistics:
    - Total samples: 1,888
    - Original features: 13
    - Engineered features: 36 (after feature engineering and selection)
    - Class distribution: Approximately 48% No Disease, 52% Disease
    - Data preprocessing: Outlier handling, SMOTE balancing, robust scaling
    
    #### Models Available:
    
    1. **Logistic Regression**: Linear model for interpretable predictions. Uses logistic function 
       to model the probability of heart disease based on input features. Fast and provides 
       insights into which features contribute most to predictions.
    
    2. **Random Forest**: Ensemble of decision trees that vote on the final prediction. Creates 
       multiple decision trees using random subsets of features and data, then averages their 
       predictions. Robust and handles non-linear relationships well.
    
    3. **HRLFM (High-Resolution Logistic-Forest Model)**: Hybrid ensemble combining Logistic 
       Regression and Random Forest. Uses weighted voting to balance linear interpretability 
       with non-linear predictive power, providing both accuracy and explainability.
    
    #### Model Performance:
    - All models achieved >75% accuracy
    - Models were trained with hyperparameter tuning
    - HRLFM provides balanced performance with high interpretability
    - Cross-validated to ensure robust performance
    """)
    
    # Display model comparison if available
    st.markdown("---")
    st.markdown("#### 📊 Model Performance Comparison")
    try:
        model_path = Path(__file__).parent.parent / 'models'
        if (model_path / 'model_comparison.csv').exists():
            comparison_df = pd.read_csv(model_path / 'model_comparison.csv')
            st.dataframe(comparison_df, use_container_width=True)
    except Exception as e:
        st.info("Model comparison data not available.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D;'>"
    "Heart Disease Predictor | Built with Streamlit, Scikit-learn & XGBoost | "
    "For educational purposes only"
    "</p>",
    unsafe_allow_html=True
)
