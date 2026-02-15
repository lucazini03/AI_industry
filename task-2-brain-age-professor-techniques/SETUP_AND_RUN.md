# Setup & Execution Guide

## Quick Start

### 1. Install the Package

From the project directory (`task-2-brain-age-professor-techniques/`):

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e .
```

**Why `-e` (editable)?**
- Makes the `brain_age_prof` module discoverable by Python
- Changes to source files are immediately reflected (no reinstall needed)
- Allows running as a module: `python -m brain_age_prof.cli`

### 2. Run the Pipeline

```bash
# Quick mode (for testing, ~5 minutes)
python -m brain_age_prof.cli --quick --output-dir outputs

# Full mode (best accuracy, ~20 minutes)
python -m brain_age_prof.cli --output-dir outputs
```

**Expected output:**
```
Stage 1: Loading data...
  X_train shape: (1212, 832)
  X_test shape: (776, 832)
  Missing values in X_train: 76902
...
Stage 10: Exporting results...

======================================================================
PIPELINE COMPLETED SUCCESSFULLY
======================================================================
Selected features: 65
Best model: random_forest
Outputs written to: .../outputs
```

---

## Why the Original Error Occurred

### Error Message
```
ModuleNotFoundError: No module named 'brain_age_prof'
```

### Root Cause
The project uses a **package layout** with code inside `src/brain_age_prof/`. Python couldn't find this module because:
1. The package wasn't installed (no `pip install`)
2. The module location wasn't in Python's search path
3. Trying to run `python -m brain_age_prof.cli` without installation fails

### Solution: Editable Installation
```bash
pip install -e .  # Installs package from pyproject.toml
```

This:
- Reads `pyproject.toml` configuration
- Locates the package at `src/brain_age_prof/`
- Adds it to Python's search path
- Allows `python -m brain_age_prof.cli` to work

---

## Project Architecture

### File Structure & Responsibilities

```
src/brain_age_prof/
├── __init__.py              Version info, package metadata
├── cli.py                   ✓ Entry point, argument parsing
├── config.py                ✓ Configuration dataclasses
├── data.py                  ✓ CSV loading, validation
├── imputation.py            ✓ Gaussian Process feature imputer
├── feature_selection.py     ✓ 4 importance measures + selection logic
├── modeling.py              ✓ Model building, cross-validation
└── pipeline.py              ✓ Orchestrates all stages
```

### Data Flow

```
Data Input (CSVs)
    ↓
[cli.py] Parse arguments
    ↓
[data.py] Load X_train, y_train, X_test
    ↓
[pipeline.py] Split data (80/20 dev/holdout)
    ↓
[imputation.py] Fit GP imputer on dev → transform all sets
    ↓
[feature_selection.py] Compute 4 importance measures
    ├─ model_importance (RF native)
    ├─ permutation_importance (holdout perf drop)
    ├─ shap_importance (Shapley values)
    └─ boruta_support (all-relevant selection)
    ↓
[feature_selection.py] Combine measures → select features
    ↓
[modeling.py] Build & CV models on selected features
    ↓
[pipeline.py] Train best model, predict test set
    ↓
Export: cv_results.csv, selected_features.csv, importance_*.csv, submission.csv
```

---

## What Each Stage Does

### Stage 1: Data Loading
- Reads X_train, y_train, X_test from CSV
- Validates that y_train has 'y' column
- Extracts and preserves sample IDs as indices

### Stage 2: Data Splitting
- Splits training data: 80% development, 20% holdout
- Development set: used to fit GP imputer & importance model
- Holdout set: used to compute permutation & SHAP importance

### Stage 3: GP Imputation
- For each feature with missing values:
  - Find top-K correlated features (by Pearson correlation)
  - Train a Gaussian Process regressor
  - Predict missing values with trained GP
- Features with <100 observed values are skipped
- **Result**: Zero missing values in all datasets

### Stage 4: Feature Importance (4 Methods)

| Method | What It Measures | Strengths |
|--------|------------------|-----------|
| **Native** | RF impurity decrease | Fast, built-in, captures high-order interactions |
| **Permutation** | Performance drop when shuffled | Model-agnostic, reflects real predictive impact |
| **SHAP** | Shapley-based feature attribution | Theoretically principled, stable across samples |
| **Boruta** | All relevant features via shadow comparison | Identifies weak but meaningful features |

### Stage 5: Feature Selection
- Takes union of:
  - Top-50 permutation importance features
  - Top-50 SHAP importance features
  - All Boruta-supported features
- **Result**: ~65 features (varies by run)
- Ensures diverse selection across methods

### Stage 6-7: Model Training & Evaluation
- Builds two models:
  - Random Forest: 500 trees (or 220 in quick mode)
  - HistGradientBoosting: 320 iterations (or 180 in quick mode)
- Cross-validates with 5-fold CV (or 3 in quick mode)
- **Metrics**: MAE, RMSE, R²
- Selects best model (usually Random Forest)

### Stage 8-9: Test Prediction
- Trains best model on full training set
- Predicts on test set
- Exports predictions as submission.csv

---

## Understanding the Output Files

### cv_results.csv
```
model,mae_mean,mae_std,rmse_mean,r2_mean
random_forest,5.117,0.115,6.868,0.495
hist_gb,5.151,0.191,6.944,0.482
```
- **mae_mean**: Average prediction error (~5 years off)
- **r2_mean**: Model explains ~50% of age variance
- Random Forest outperforms boosting on this task

### selected_features.csv
```
feature
x0
x113
x115
...
```
65 features selected from original 832

### importance_*.csv
**importance_native.csv**: RF impurity decrease
**importance_permutation.csv**: Performance drop when shuffled
**importance_shap.csv**: SHAP mean absolute values
**importance_boruta_mask.csv**: Boolean (1=important, 0=not)

### submission.csv
```
id,y
0,72.5
1,65.3
...
```
Final predictions on test set

---

## Quick vs. Full Mode

| Aspect | Quick | Full |
|--------|-------|------|
| **GP Features** | 40 | 80 |
| **Aux Features/GP** | 12 | 20 |
| **RF Trees** | 220 | 500 |
| **Boosting Iters** | 180 | 320 |
| **CV Splits** | 3 | 5 |
| **Permutation Repeats** | 8 | 20 |
| **SHAP Sample Size** | 320 | 700 |
| **Boruta Iters** | 20 | 40 |
| **Runtime** | ~5 min | ~20 min |
| **Accuracy** | Baseline | Better |

---

## Customization

### Change Data Directory
```bash
python -m brain_age_prof.cli \
  --data-dir /custom/path/to/data \
  --output-dir outputs
```

### Change Output Directory
```bash
python -m brain_age_prof.cli \
  --output-dir /custom/output/path
```

### Change Configuration in Code

Edit `cli.py` to modify defaults:
```python
imputation_cfg = ImputationConfig(
    max_gp_features=100,  # More features for GP
    n_aux_features=25,     # More predictors per GP
    min_observed=80,       # Lower threshold
)
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'brain_age_prof'`
```bash
# Solution: Install package
pip install -e .
```

### Issue: `FileNotFoundError: X_train.csv not found`
```bash
# Check data directory exists
ls ../task-1-brain-age-prediction/

# Specify correct path
python -m brain_age_prof.cli --data-dir ../task-1-brain-age-prediction
```

### Issue: Process is very slow
```bash
# Use quick mode
python -m brain_age_prof.cli --quick
```

### Issue: Out of memory
```bash
# Reduce GP features in cli.py
max_gp_features=30  # Instead of 80
```

### Issue: SHAP warnings/errors
- These are normal (harmless convergence warnings from GPs)
- Pipeline continues and produces valid results

---

## Key Technologies & Concepts

### Gaussian Processes
- Non-parametric Bayesian regression
- Captures uncertainty in predictions
- Good for high-dimensional tabular data with missing values
- See: `03-sf-gp/3. Gaussian Processes.ipynb`

### Permutation Importance
- Measures drop in model performance when feature is shuffled
- Model-agnostic (works with any regressor)
- More stable than native importances
- See: `06-at/4. Additive Feature Attribution.ipynb`

### SHAP Values
- Game-theoretic approach to feature attribution
- Explains individual predictions (local) and model behavior (global)
- Theoretically principled (satisfies Shapley axioms)
- See: `06-at/4. Additive Feature Attribution.ipynb`

### Boruta Algorithm
- All-relevant feature selection (not minimal)
- Compares real features against random "shadow" features
- Identifies both strong and weak but important features
- See: `06-at/5. All Relevant Feature Selection.ipynb`

---

## Performance Expectations

On the brain age dataset (~1200 training samples, 832 features):
- **Quick mode**: MAE ~5 years, R² ~0.49
- **Full mode**: MAE ~4.8-5.0 years, R² ~0.50-0.52
- **Baseline** (task-1): MAE ~5.2 years

Improvement comes from:
1. Better imputation (GPs vs. simple mean)
2. Better feature selection (ensemble of methods)
3. Cross-validated model selection

---

## Next Steps

1. **Examine importance files**: Which features matter most?
2. **Inspect selected features**: Compare across importance methods
3. **Try full mode**: Better results for production
4. **Tune hyperparameters**: Edit config in `cli.py`
5. **Compare with task-1**: See how techniques from 03-sf-gp & 06-at help

