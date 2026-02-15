"""
Main ML pipeline orchestrating all stages: imputation, feature selection, and modeling.

Pipeline Flow:
  1. Load data (X_train, y_train, X_test)
  2. Split training data into dev/holdout sets for feature importance computation
  3. Impute missing features using Gaussian Processes (fitted on dev set)
  4. Build importance model and compute 4 types of feature importance:
     - Native RF importances
     - Permutation importance (on holdout set)
     - SHAP importance
     - Boruta support (on full training set)
  5. Combine importance measures to select final feature set
  6. Train models on selected features
  7. Cross-validate to find best model
  8. Make test predictions with best model
  9. Export results (CV scores, selected features, importances, predictions)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .config import ImputationConfig, SelectionConfig, TrainingConfig
from .data import load_brain_age_data
from .feature_selection import (
    boruta_support,
    build_selected_features,
    model_importance,
    permutation_importance_series,
    shap_importance_series,
)
from .imputation import GaussianProcessFeatureImputer
from .modeling import build_models, cross_validate_models


def run_pipeline(
    data_dir: str,
    output_dir: str,
    imputation_cfg: ImputationConfig,
    selection_cfg: SelectionConfig,
    training_cfg: TrainingConfig,
) -> None:
    """
    Execute the full brain age prediction pipeline.
    
    Args:
        data_dir (str): Path to directory with X_train.csv, y_train.csv, X_test.csv
        output_dir (str): Path to output directory for results
        imputation_cfg (ImputationConfig): Configuration for GP imputation
        selection_cfg (SelectionConfig): Configuration for feature selection
        training_cfg (TrainingConfig): Configuration for model training
    
    Outputs:
        - cv_results.csv: Cross-validation metrics for each model
        - selected_features.csv: List of selected features
        - importance_native.csv: Model native importances
        - importance_permutation.csv: Permutation importances
        - importance_shap.csv: SHAP importances
        - importance_boruta_mask.csv: Boruta boolean support
        - submission.csv: Final predictions on test set
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # STAGE 1: LOAD DATA
    # ============================================================================
    print("Stage 1: Loading data...")
    X_train, y_train, X_test = load_brain_age_data(data_dir)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Missing values in X_train: {X_train.isna().sum().sum()}")

    # ============================================================================
    # STAGE 2: SPLIT DATA FOR IMPORTANCE COMPUTATION
    # ============================================================================
    # Split training data: 80% dev (for imputation & importance), 20% holdout (for validation)
    print("\nStage 2: Splitting training data (80% dev, 20% holdout)...")
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=training_cfg.random_state,
    )
    print(f"  X_dev shape: {X_dev.shape}")
    print(f"  X_hold shape: {X_hold.shape}")

    # ============================================================================
    # STAGE 3: IMPUTE MISSING VALUES USING GAUSSIAN PROCESSES
    # ============================================================================
    print("\nStage 3: Imputing missing features with Gaussian Processes...")
    gp_imputer = GaussianProcessFeatureImputer(
        max_gp_features=imputation_cfg.max_gp_features,
        n_aux_features=imputation_cfg.n_aux_features,
        min_observed=imputation_cfg.min_observed,
        random_state=training_cfg.random_state,
    )
    
    # Fit on dev set, transform all sets
    X_dev_imp = gp_imputer.fit_transform(X_dev)
    X_hold_imp = gp_imputer.transform(X_hold)
    X_train_imp = gp_imputer.transform(X_train)
    X_test_imp = gp_imputer.transform(X_test)
    print(f"  Trained {len(gp_imputer.models_)} GP models")
    print(f"  Remaining missing values: {X_train_imp.isna().sum().sum()}")

    # ============================================================================
    # STAGE 4: COMPUTE FEATURE IMPORTANCE (4 METHODS)
    # ============================================================================
    print("\nStage 4: Computing feature importances (4 methods)...")
    
    # Build Random Forest on dev set for importance computation
    importance_model = RandomForestRegressor(
        n_estimators=250 if training_cfg.quick else 450,
        random_state=training_cfg.random_state,
        n_jobs=-1,
    )
    importance_model.fit(X_dev_imp, y_dev)

    # 4a. Native RF importances
    print("  - Computing native RF importances...")
    native_imp = model_importance(importance_model, list(X_dev_imp.columns))
    
    # 4b. Permutation importance
    print("  - Computing permutation importances...")
    perm_imp = permutation_importance_series(
        importance_model,
        X_hold_imp,
        y_hold,
        n_repeats=8 if training_cfg.quick else 20,
        random_state=training_cfg.random_state,
    )

    # 4c. SHAP importance
    print("  - Computing SHAP importances...")
    shap_sample = X_hold_imp.sample(
        n=min(320 if training_cfg.quick else 700, len(X_hold_imp)),
        random_state=training_cfg.random_state,
    )
    shap_imp = shap_importance_series(importance_model, shap_sample)

    # 4d. Boruta support
    print("  - Running Boruta algorithm...")
    boruta_mask = boruta_support(
        X_dev_imp,
        y_dev,
        random_state=training_cfg.random_state,
        max_iter=selection_cfg.boruta_max_iter if not training_cfg.quick else max(20, selection_cfg.boruta_max_iter // 2),
    )
    print(f"  - Boruta selected {boruta_mask.sum()} features")

    # ============================================================================
    # STAGE 5: COMBINE IMPORTANCE MEASURES INTO FINAL FEATURE SET
    # ============================================================================
    print("\nStage 5: Selecting final feature set...")
    selected_features = build_selected_features(
        perm_imp=perm_imp,
        shap_imp=shap_imp,
        boruta_mask=boruta_mask,
        top_k_perm=selection_cfg.top_k_perm,
        top_k_shap=selection_cfg.top_k_shap,
    )
    print(f"  Selected {len(selected_features)} features (union of top-K + Boruta)")

    # ============================================================================
    # STAGE 6: PREPARE DATA FOR MODELING
    # ============================================================================
    print("\nStage 6: Preparing selected features for modeling...")
    X_train_sel = X_train_imp[selected_features]
    X_test_sel = X_test_imp[selected_features]

    # ============================================================================
    # STAGE 7: CROSS-VALIDATE MODELS
    # ============================================================================
    print("\nStage 7: Cross-validating models...")
    models = build_models(training_cfg.quick, training_cfg.random_state)
    cv_results = cross_validate_models(
        models=models,
        X=X_train_sel,
        y=y_train,
        n_splits=training_cfg.cv_splits,
        random_state=training_cfg.random_state,
    )
    print("  CV Results:")
    print(cv_results.to_string(index=False))

    # ============================================================================
    # STAGE 8: TRAIN BEST MODEL ON FULL TRAINING SET
    # ============================================================================
    print("\nStage 8: Training best model on full training set...")
    best_name = cv_results.iloc[0]["model"]
    best_model = models[best_name]
    best_model.fit(X_train_sel, y_train)
    print(f"  Best model: {best_name}")

    # ============================================================================
    # STAGE 9: MAKE TEST PREDICTIONS
    # ============================================================================
    print("\nStage 9: Making predictions on test set...")
    y_pred_test = best_model.predict(X_test_sel)
    submission = pd.DataFrame({"id": X_test.index, "y": y_pred_test})
    submission = submission.set_index("id")
    print(f"  Predictions: {len(y_pred_test)} samples")
    print(f"  Mean predicted age: {y_pred_test.mean():.2f} years")

    # ============================================================================
    # STAGE 10: EXPORT RESULTS
    # ============================================================================
    print("\nStage 10: Exporting results...")
    cv_results.to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(out / "selected_features.csv", index=False)
    native_imp.to_csv(out / "importance_native.csv", header=["importance"])
    perm_imp.to_csv(out / "importance_permutation.csv", header=["importance"])
    shap_imp.to_csv(out / "importance_shap.csv", header=["importance"])
    boruta_mask.to_csv(out / "importance_boruta_mask.csv", header=["selected"])
    submission.to_csv(out / "submission.csv")

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Selected features: {len(selected_features)}")
    print(f"Best model: {best_name}")
    print(f"Outputs written to: {out.resolve()}")
    print("="*70)
