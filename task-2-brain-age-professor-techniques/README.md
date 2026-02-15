# Brain Age Prediction with GP Imputation + Explainable Feature Selection

This project reproduces the brain-age prediction task on the same dataset used in `task-1-brain-age-prediction`, while explicitly applying advanced techniques covered in:

- **03-sf-gp**: Gaussian Process-based imputation for missing values
- **06-at**: Feature importance, explainability, and selection (Permutation Importance, SHAP, Boruta)

## What This Project Does

The pipeline orchestrates a complete machine learning workflow:

1. **Data Loading**: Reads `X_train.csv`, `y_train.csv`, `X_test.csv`
2. **Data Splitting**: Splits training data into development (80%) and holdout (20%) sets
3. **Imputation**: Uses Gaussian Process feature imputation for handling missing values
   - Trains one GP per feature with missing values
   - Uses top-K correlated features as predictors
   - Predicts missing values with trained GPs
4. **Feature Importance**: Computes 4 complementary importance measures:
   - **Native Importance**: Random Forest feature importances (impurity decrease)
   - **Permutation Importance**: Drop in performance when feature is shuffled
   - **SHAP Importance**: Game-theoretic Shapley values for additive attribution
   - **Boruta Support**: All-relevant feature selection algorithm
5. **Feature Selection**: Combines all measures into final feature set (union of top-K + Boruta)
6. **Model Training**: Builds and cross-validates multiple regressors
   - Random Forest
   - HistGradientBoosting
7. **Evaluation**: Computes MAE, RMSE, R² across K-fold CV
8. **Export**: Saves results and predictions

## Dataset Location

By default, the pipeline expects data in:

```
../task-1-brain-age-prediction/
  ├── X_train.csv       # Training features (may have NaN)
  ├── y_train.csv       # Training targets (must have 'y' column)
  └── X_test.csv        # Test features
```

Override with `--data-dir` argument.

## Installation & Setup

### 1. Install Dependencies

From the project root (`task-2-brain-age-professor-techniques/`):

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with dependencies
pip install -e .
```

The `-e` (editable) flag installs the package so Python can find the `brain_age_prof` module.

### 2. Verify Installation

```bash
# Check that the package is installed
python -c "import brain_age_prof; print(brain_age_prof.__version__)"

# Test the CLI entry point
python -m brain_age_prof.cli --help
```

## Running the Pipeline

### Full Run (Best Accuracy)

```bash
python -m brain_age_prof.cli \
  --data-dir ../task-1-brain-age-prediction \
  --output-dir outputs
```

This runs with default settings:
- 80 features for GP imputation
- 5 cross-validation splits
- Full model complexity (500 trees for RF, 320 iterations for boosting)

### Quick Mode (For Development/Testing)

```bash
python -m brain_age_prof.cli \
  --data-dir ../task-1-brain-age-prediction \
  --output-dir outputs \
  --quick
```

Quick mode uses:
- 40 features for GP imputation
- 3 cross-validation splits
- Lighter models (220 trees, 180 iterations)
- Faster feature importance computation

### Custom Output Directory

```bash
python -m brain_age_prof.cli \
  --data-dir ../task-1-brain-age-prediction \
  --output-dir my_results \
  --quick
```

## Output Files

After running, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `cv_results.csv` | Cross-validation metrics (MAE, RMSE, R²) for each model |
| `selected_features.csv` | List of selected feature names (union of importance measures) |
| `importance_native.csv` | Native Random Forest importances |
| `importance_permutation.csv` | Permutation importances (feature drop in performance) |
| `importance_shap.csv` | SHAP mean absolute values per feature |
| `importance_boruta_mask.csv` | Boolean support vector from Boruta algorithm |
| `submission.csv` | Final predictions on test set (id, y columns) |

## Project Structure

```
task-2-brain-age-professor-techniques/
├── pyproject.toml                     # Package configuration
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── src/brain_age_prof/
    ├── __init__.py                    # Package metadata
    ├── cli.py                         # Command-line interface & argument parsing
    ├── config.py                      # Configuration dataclasses
    ├── data.py                        # Data loading utilities
    ├── imputation.py                  # Gaussian Process feature imputer
    ├── feature_selection.py           # Importance computation & feature selection
    ├── modeling.py                    # Model building & cross-validation
    └── pipeline.py                    # Main pipeline orchestration
```

### File Descriptions

- **cli.py**: Entry point. Parses command-line arguments and configures pipeline stages.
- **config.py**: Dataclasses for `ImputationConfig`, `SelectionConfig`, `TrainingConfig`.
- **data.py**: Loads CSV files, validates columns, manages indices.
- **imputation.py**: `GaussianProcessFeatureImputer` class. Trains one GP per feature with missing values.
- **feature_selection.py**: Functions for computing 4 types of importance + combining them.
- **modeling.py**: Builds regression models and performs cross-validation.
- **pipeline.py**: Orchestrates all stages in sequence, handles data flow.

## How to Troubleshoot

### Issue: `ModuleNotFoundError: No module named 'brain_age_prof'`

**Solution**: Ensure you've installed the package in editable mode:

```bash
pip install -e .
```

### Issue: Data files not found

**Solution**: Check that data directory exists and contains the CSV files:

```bash
ls -la ../task-1-brain-age-prediction/
# Should show: X_train.csv, y_train.csv, X_test.csv
```

### Issue: Out of memory / Very slow

**Solution**: Use quick mode to reduce memory usage and runtime:

```bash
python -m brain_age_prof.cli --quick --output-dir outputs
```

## Key Techniques Applied

### 1. Gaussian Process Imputation (from 03-sf-gp)
- Non-parametric approach using GP regression
- Captures relationships between features
- Better than simple mean/median for structured data

### 2. Permutation Importance (from 06-at)
- Model-agnostic feature importance
- Measures drop in performance when feature is shuffled
- Reflects actual predictive impact

### 3. SHAP Values (from 06-at)
- Game-theoretic approach to feature attribution
- Explains model predictions with Shapley values
- Provides both global and local explanations

### 4. Boruta Algorithm (from 06-at)
- All-relevant feature selection (vs. minimal set)
- Compares real features against random shadows
- Identifies both strong and weak but relevant features

### 5. Feature Selection Strategy
- Union of top-K (permutation) + top-K (SHAP) + Boruta support
- Balances different perspectives on importance
- Avoids over-reliance on any single method

## Performance Metrics

The pipeline reports:
- **MAE**: Mean Absolute Error (in original units, years)
- **RMSE**: Root Mean Squared Error (penalizes larger errors)
- **R²**: Coefficient of Determination (proportion of variance explained, [0, 1])

## References

- **Gaussian Processes**: See notebook 03-sf-gp/3. Gaussian Processes.ipynb
- **Feature Importance**: See notebook 06-at/4. Additive Feature Attribution.ipynb
- **Feature Selection**: See notebook 06-at/5. All Relevant Feature Selection.ipynb

## Notes

- The GP imputer is computationally efficient (one GP per feature, not global)
- Boruta and SHAP can be slow on very large datasets; use `--quick` mode first
- Results are deterministic if random_state is fixed (default: 42)
- All operations use parallel processing (`n_jobs=-1`) where available
