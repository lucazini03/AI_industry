"""
Command-line interface for brain age prediction pipeline.

This module provides the entry point for the full ML pipeline:
  1. Load brain imaging and demographic data
  2. Impute missing features using Gaussian Processes
  3. Select important features via permutation importance, SHAP, and Boruta
  4. Train and cross-validate regression models (Random Forest, HistGradientBoosting)
  5. Export results (selected features, importances, predictions, CV scores)

Usage:
    python -m brain_age_prof.cli --data-dir ../task-1-brain-age-prediction --output-dir outputs
    python -m brain_age_prof.cli --data-dir ../task-1-brain-age-prediction --output-dir outputs --quick
"""
from __future__ import annotations

import argparse

from .config import ImputationConfig, SelectionConfig, TrainingConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with data_dir, output_dir, and quick flags
    """
    parser = argparse.ArgumentParser(
        description="Brain age prediction with GP imputation + Boruta/SHAP selection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../task-1-brain-age-prediction",
        help="Directory with X_train.csv, y_train.csv, X_test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where artifacts and submission will be written",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a faster, lighter setup (fewer features, iterations, CV splits)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point. Reads arguments, configures pipeline stages, and runs full pipeline.
    
    Configuration is adjusted based on --quick flag:
      - Quick mode: fewer GP features, fewer auxiliary features, fewer Boruta iterations
      - Full mode: more features and iterations for better accuracy
    """
    args = parse_args()

    # Imputation configuration: controls how Gaussian Processes fill missing values
    imputation_cfg = ImputationConfig(
        max_gp_features=40 if args.quick else 80,  # Max features to build GPs for
        n_aux_features=12 if args.quick else 20,   # Number of correlated predictors per GP
        min_observed=100,                           # Min non-missing values required to train GP
    )
    
    # Feature selection configuration: controls thresholds for importance-based filtering
    selection_cfg = SelectionConfig(
        top_k_perm=50 if args.quick else 80,   # Keep top-K permutation importance features
        top_k_shap=50 if args.quick else 80,   # Keep top-K SHAP importance features
        boruta_max_iter=20 if args.quick else 40,  # Boruta algorithm iterations
    )
    
    # Training configuration: controls model building and validation
    training_cfg = TrainingConfig(
        cv_splits=3 if args.quick else 5,  # K-fold cross-validation splits
        random_state=42,                    # Seed for reproducibility
        quick=args.quick,                   # Flag for model complexity
    )

    # Run the full pipeline
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        imputation_cfg=imputation_cfg,
        selection_cfg=selection_cfg,
        training_cfg=training_cfg,
    )


if __name__ == "__main__":
    main()
