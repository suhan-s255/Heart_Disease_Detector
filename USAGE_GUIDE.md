# HRLFM Pipeline - Quick Usage Guide

This guide shows you how to use the HRLFM (High-Resolution Logistic-Forest Model) pipeline for heart disease prediction.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- 5-10 minutes of time (for pipeline execution)

## Quick Start (Recommended)

### Step 1: Clone and Navigate

```bash
git clone https://github.com/Sarthaka-Mitra/heart-disease-detector.git
cd heart-disease-detector
```

### Step 2: Run the Automated Setup Script

```bash
bash run_hrlfm_pipeline.sh
```

**What this script does:**
1. Checks if Python 3.8+ is installed
2. Creates a virtual environment (`venv_hrlfm`)
3. Installs all required dependencies
4. Verifies the dataset exists
5. Runs the complete HRLFM training pipeline
6. Saves 8 trained models to the `models/` directory
7. Generates visualizations and performance metrics

**Expected output:**
- 8 trained models (`.pkl` files)
- Performance comparison CSV
- Feature importance plots
- Confusion matrices
- ROC curves
- SHAP and LIME explanations

**Expected runtime:** 5-10 minutes depending on your hardware

### Step 3: Run the Web Application

After the pipeline completes:

```bash
# Activate the virtual environment
source venv_hrlfm/bin/activate  # On Windows: venv_hrlfm\Scripts\activate

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 4: Make Predictions

In the web app:
1. Select a model from the dropdown (8 models available)
2. Enter patient information:
   - Age (29-77)
   - Sex (Male/Female)
   - Chest pain type (0-4)
   - Blood pressure (94-200 mm Hg)
   - Cholesterol (126-564 mg/dl)
   - And other clinical measurements
3. Click "Predict Heart Disease Risk"
4. View results with probability breakdown

### Step 5: Explore Performance Metrics

Use the app tabs to:
- Compare model performances
- View confusion matrices
- Analyze ROC curves
- Explore feature importance

## Manual Setup (Alternative)

If you prefer more control:

### 1. Create Virtual Environment

```bash
python3 -m venv venv_hrlfm
source venv_hrlfm/bin/activate  # On Windows: venv_hrlfm\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the Pipeline

```bash
python train_hrlfm_pipeline.py
```

### 4. Run the App

```bash
streamlit run app/streamlit_app.py
```

## Understanding the Pipeline

### What is HRLFM?

**High-Resolution Logistic-Forest Model** is a hybrid machine learning approach that combines:
- **Logistic Regression**: Captures linear relationships
- **Random Forest**: Handles non-linear patterns
- **XGBoost**: Adds gradient boosting power

The pipeline trains 8 models:
1. Logistic Regression
2. Random Forest - Best: 97.88% accuracy
3. XGBoost
4. LightGBM
5. SVM
6. Voting Ensemble
7. Stacking Ensemble
8. HRLFM

### Pipeline Steps

1. **Data Loading**: Load 1,888 patient records
2. **Missing Value Handling**: Median/mode imputation
3. **Outlier Detection**: IQR-based capping
4. **Feature Engineering**: Create 23 new features
   - 8 domain-specific features
   - 15 polynomial features
5. **Preprocessing**: RobustScaler + SMOTE
6. **Feature Selection**: SelectKBest + SelectFromModel
7. **Model Training**: Train 5 baseline models
8. **Ensemble Creation**: Voting + Stacking
9. **HRLFM Training**: Hybrid meta-model
10. **Evaluation**: Metrics + Visualizations
11. **Model Persistence**: Save all models

### Performance Metrics

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 97.88% | 0.9978 |
| LightGBM | 97.62% | 0.9984 |
| Voting Ensemble | 97.62% | 0.9982 |
| XGBoost | 96.83% | 0.9970 |
| HRLFM | 96.30% | 0.9960 |

## Troubleshooting

### Problem: "ModuleNotFoundError"

**Solution:**
```bash
source venv_hrlfm/bin/activate
pip install -r requirements.txt
```

### Problem: "Dataset not found"

**Solution:**
Ensure `data/cleaned_merged_heart_dataset.csv` exists. If not, re-clone the repository.

### Problem: "No models found in Streamlit app"

**Solution:**
Train the models first:
```bash
python train_hrlfm_pipeline.py
```

### Problem: "Permission denied" for bash script

**Solution:**
```bash
chmod +x run_hrlfm_pipeline.sh
bash run_hrlfm_pipeline.sh
```

### Problem: Python version too old

**Solution:**
The pipeline requires Python 3.8+. Upgrade Python:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.8

# macOS
brew install python@3.8
```

## Running Tests

To verify everything works:

```bash
source venv_hrlfm/bin/activate
python -m unittest discover -s tests -v
```

## File Structure

```
heart-disease-detector/
├── run_hrlfm_pipeline.sh       # Automated setup script
├── train_hrlfm_pipeline.py     # Pipeline training script
├── app/
│   └── streamlit_app.py        # Web application
├── data/
│   └── cleaned_merged_heart_dataset.csv  # Dataset (1,888 records)
├── models/                     # Generated after training
│   ├── *.pkl                   # Trained models
│   └── *.png                   # Visualizations
├── tests/
│   └── test_app.py             # Unit tests
└── requirements.txt            # Dependencies
```

## Next Steps

1. **Explore the App**: Try different models and input values
2. **Read Documentation**: Check `HRLFM_PIPELINE.md` for details
3. **Experiment**: Modify hyperparameters in `train_hrlfm_pipeline.py`
4. **Deploy**: Host the app on Streamlit Cloud, Heroku, or AWS

## Getting Help

- **Documentation**: See `HRLFM_PIPELINE.md` for complete details
- **Issues**: Open an issue on GitHub
- **Contributing**: See `CONTRIBUTING.md`

## Important Notes

**Medical Disclaimer**: This tool is for educational and research purposes only. Do not use it as a substitute for professional medical advice, diagnosis, or treatment.

**Model Performance**: All models achieve >85% accuracy, with Random Forest achieving 97.88%

**Privacy**: All processing happens locally. No data is sent to external servers.

---

**Built using Python, Scikit-learn, XGBoost, LightGBM, and Streamlit**
