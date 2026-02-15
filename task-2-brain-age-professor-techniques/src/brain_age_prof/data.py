"""
Data loading utilities for brain age prediction.

This module handles loading and preprocessing of training and test data
from the task-1-brain-age-prediction dataset.
"""
from pathlib import Path

import pandas as pd


def load_brain_age_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load brain age prediction dataset from CSV files.
    
    Expected files in data_dir:
      - X_train.csv: Training features (may contain 'id' column)
      - y_train.csv: Training target (must contain 'y' column)
      - X_test.csv: Test features (may contain 'id' column)
    
    The function:
      1. Loads all three CSV files
      2. Validates that y_train has a 'y' column
      3. Removes 'id' columns from X (if present) and uses them as DataFrame indices
      4. Returns feature DataFrames and target Series with aligned indices
    
    Args:
        data_dir (str | Path): Directory containing the CSV files.
    
    Returns:
        tuple: (X_train, y_train, X_test)
          - X_train (pd.DataFrame): Training features, indexed by sample ID if available
          - y_train (pd.Series): Training targets (age in years)
          - X_test (pd.DataFrame): Test features, indexed by sample ID if available
    
    Raises:
        ValueError: If y_train.csv does not contain a 'y' column.
    """
    data_dir = Path(data_dir)

    # Load CSV files
    x_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    x_test = pd.read_csv(data_dir / "X_test.csv")

    # Validate target column exists
    if "y" not in y_train.columns:
        raise ValueError("Expected target column `y` inside y_train.csv")

    # Make copies to avoid modifying originals
    x_train = x_train.copy()
    x_test = x_test.copy()

    # Extract ID columns if present and use as index
    x_train_ids = x_train["id"].copy() if "id" in x_train.columns else None
    x_test_ids = x_test["id"].copy() if "id" in x_test.columns else None

    # Keep only feature columns (drop 'id')
    feature_cols = [c for c in x_train.columns if c != "id"]
    x_train = x_train[feature_cols]
    x_test = x_test[feature_cols]

    # Extract target as Series and convert to float
    y = y_train["y"].astype(float)

    # Set indices if IDs were available
    if x_train_ids is not None:
        x_train.index = x_train_ids
    if x_test_ids is not None:
        x_test.index = x_test_ids

    return x_train, y, x_test
