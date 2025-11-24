# Synthetic Data Generation for Heart Disease Dataset

This guide explains how to generate synthetic data to augment the heart disease dataset from 1,888 rows to 8,768 rows using CTGAN (Conditional Tabular GAN).

## Overview

The synthetic data generation pipeline:
1. Trains a CTGAN model on the cleaned heart disease dataset (`cleaned_merged_heart_dataset.csv`)
2. Generates 6,880 synthetic rows that are fully compatible with the original data
3. Combines original and synthetic data, removes duplicates
4. Evaluates synthetic data quality and ML model performance

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- CPU or GPU (CPU version instructions provided)

## Installation

### Step 1: Create and Activate Virtual Environment

**For zsh (macOS/Linux):**
```zsh
# Create virtual environment
python3 -m venv venv_synthesis

# Activate virtual environment
source venv_synthesis/bin/activate
```

**For bash (Linux):**
```bash
# Create virtual environment
python3 -m venv venv_synthesis

# Activate virtual environment
source venv_synthesis/bin/activate
```

**For Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv_synthesis

# Activate virtual environment
.\venv_synthesis\Scripts\Activate.ps1
```

**For Conda (alternative):**
```bash
# Create conda environment
conda create -n synthesis python=3.9

# Activate conda environment
conda activate synthesis
```

### Step 2: Install PyTorch (CPU Version)

Install PyTorch CPU version first to avoid large GPU-specific dependencies:

```zsh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Note:** This installs ~200MB instead of ~2GB for GPU versions.

### Step 3: Install Synthesis Dependencies

```zsh
pip install -r requirements-synthesis.txt
```

This installs:
- `ctgan` - Conditional Tabular GAN for synthetic data generation
- `sdv` - Synthetic Data Vault (optional, provides additional features)
- `sdmetrics` - Quality evaluation metrics
- `pandas`, `numpy`, `scikit-learn` - Core data science libraries

### Step 4: Verify Installation

```zsh
python -c "import torch; import ctgan; print('Installation successful!')"
```

## Usage

### Generate Synthetic Data

Run the generation script to create the augmented dataset:

```zsh
python scripts/generate_synthetic_compatible.py
```

**What it does:**
- Loads `data/cleaned_merged_heart_dataset.csv` (1,888 rows)
- Trains a CTGAN model (takes ~5-10 minutes on CPU)
- Generates 6,880 synthetic rows
- Ensures exact column names, order, and data types match original
- Removes duplicate rows
- Saves to `data/synthetic_augmented_heart_dataset.csv` (8,768 rows)

**Expected output:**
```
========================================
SYNTHETIC DATA GENERATION FOR HEART DISEASE DATASET
========================================
✓ PyTorch 2.x.x found
✓ CTGAN found
✓ Loaded 1888 rows, 14 columns
Target total rows: 8768
Synthetic rows to generate: 6880
Training CTGAN model...
✓ Generated 6880 synthetic rows
✓ Final augmented dataset: 8768 rows
```

### Evaluate Synthetic Data Quality

After generating synthetic data, evaluate its quality and ML performance:

```zsh
python scripts/evaluate_synthetic_and_model.py
```

**What it does:**
- Computes SDV quality metrics (overall score and property scores)
- Trains Random Forest classifier on both datasets
- Compares accuracy, precision, recall, F1, and AUC
- Saves results to `reports/synthetic_evaluation.json` and `.txt`

**Expected output:**
```
========================================
DATA QUALITY EVALUATION
========================================
✓ Overall Quality Score: 0.85

MACHINE LEARNING EVALUATION
========================================
✓ Baseline accuracy: 0.850
✓ Augmented accuracy: 0.847
✓ Difference: -0.003 (-0.35%)
✓ PASS: All metrics within ±3% absolute difference
```

## Output Files

### Generated Data
- **`data/synthetic_augmented_heart_dataset.csv`** - Augmented dataset (8,768 rows)
  - Contains original 1,888 rows + 6,880 synthetic rows
  - Same columns and data types as original
  - Ready for ML model training

### Evaluation Reports
- **`reports/synthetic_evaluation.json`** - Machine-readable results
- **`reports/synthetic_evaluation.txt`** - Human-readable summary

## Acceptance Criteria

✅ **Row Count:** Output file contains exactly 8,768 rows  
✅ **Column Compatibility:** Same column names and order as original  
✅ **Data Type Compatibility:** All columns have matching data types  
✅ **Quality Score:** SDV quality score ≥ 0.70 (target)  
✅ **ML Performance:** Metrics within ±3% of baseline  

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues installing PyTorch:

1. **For CPU-only (recommended for this task):**
   ```zsh
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **For CUDA 11.8 (if you have a compatible GPU):**
   ```zsh
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Check PyTorch website for other versions:**  
   https://pytorch.org/get-started/locally/

### SDV Import Failures

If SDV fails to import, the script automatically falls back to standalone CTGAN:
```
✗ SDV not found - quality evaluation will be skipped
✓ CTGAN found - will use standalone CTGAN
```

This is expected and the script will still work correctly.

### Out of Memory Errors

If you encounter OOM errors during training:

1. Reduce the number of epochs in the script (edit `epochs=300` to `epochs=100`)
2. Use a smaller batch size
3. Close other applications to free up memory

### Slow Training

CTGAN training on CPU typically takes 5-15 minutes. To speed up:

1. Use a GPU if available (install CUDA-enabled PyTorch)
2. Reduce epochs (but may affect quality)
3. Use a machine with more CPU cores

## Advanced Usage

### Adjusting Target Row Count

To generate a different number of rows, edit `generate_synthetic_compatible.py`:

```python
TARGET_TOTAL_ROWS = 8768  # Change this value
```

### Customizing CTGAN Parameters

Edit the synthesizer configuration in the generation script:

```python
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,      # Training iterations (higher = better quality, slower)
    verbose=True,    # Show training progress
    cuda=False       # Use CPU (change to True for GPU)
)
```

### Using with Existing Training Pipeline

The synthetic augmented dataset can be used as a drop-in replacement for the cleaned dataset:

```python
# Instead of:
df = pd.read_csv('data/cleaned_merged_heart_dataset.csv')

# Use:
df = pd.read_csv('data/synthetic_augmented_heart_dataset.csv')
```

Then run the existing training pipeline:
```zsh
python train_hrlfm_pipeline.py
```

## Technical Details

### CTGAN Model

CTGAN (Conditional Tabular GAN) is a generative adversarial network specifically designed for tabular data:

- **Generator:** Creates synthetic samples
- **Discriminator:** Distinguishes real from synthetic
- **Conditional Training:** Preserves relationships between features
- **Mode-specific Normalization:** Handles multi-modal distributions

### Quality Metrics

The evaluation uses SDMetrics to assess:

1. **Column Shapes:** Distribution similarity for each column
2. **Column Pair Trends:** Correlation preservation between features
3. **Overall Score:** Weighted average of all metrics

### ML Evaluation

Random Forest classifier with fixed hyperparameters:
- 100 trees
- Max depth: 10
- Random state: 42 (for reproducibility)
- 80/20 train/test split

## References

- [CTGAN Paper](https://arxiv.org/abs/1907.00503)
- [SDV Documentation](https://docs.sdv.dev/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Support

For issues or questions:
1. Check this README and troubleshooting section
2. Review script output for error messages
3. Open an issue on GitHub with:
   - Python version
   - Operating system
   - Complete error message
   - Steps to reproduce
