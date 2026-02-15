"""
Model training and cross-validation.

Builds and evaluates two regression models:
  - Random Forest: Fast, robust, captures non-linear relationships
  - HistGradientBoosting: Often more accurate, handles complex patterns
  
Cross-validates both to find the best model for final predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def build_models(quick: bool, random_state: int) -> dict[str, object]:
    """
    Build regression models with hyperparameters tuned for either quick or full mode.
    
    Quick mode: Lighter models for faster experiments / development
    Full mode: More powerful models for best accuracy
    
    Args:
        quick (bool): If True, use faster configurations.
        random_state (int): Seed for reproducibility.
    
    Returns:
        dict: Dictionary mapping model names to fitted sklearn regressors.
    """
    if quick:
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=220,
                random_state=random_state,
                n_jobs=-1,
            ),
            "hist_gb": HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.06,
                random_state=random_state,
            ),
        }

    return {
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=320,
            learning_rate=0.04,
            max_depth=8,
            random_state=random_state,
        ),
        
    }


@dataclass
class CVResult:
    """
    Cross-validation result for a single model.
    
    Attributes:
        model_name (str): Name of the model (e.g., "random_forest").
        mae_mean (float): Mean absolute error across CV folds.
        mae_std (float): Standard deviation of MAE across folds.
        rmse_mean (float): Root mean squared error across folds.
        r2_mean (float): R² score (coefficient of determination) across folds.
    """
    model_name: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    r2_mean: float


def cross_validate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cross-validate all models using K-fold strategy and report metrics.
    
    Metrics computed per fold:
      - MAE (Mean Absolute Error): Average prediction error in original units
      - RMSE (Root Mean Squared Error): Penalizes larger errors more heavily
      - R² (Coefficient of Determination): Proportion of variance explained [0-1]
    
    Args:
        models (dict): Mapping of model names to sklearn regressors.
        X (pd.DataFrame): Training features.
        y (pd.Series): Training targets.
        n_splits (int): Number of K-fold splits.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: Results sorted by MAE (best first), with columns:
          - model: Name of the model
          - mae_mean: Mean absolute error
          - mae_std: Standard deviation of MAE
          - rmse_mean: Root mean squared error
          - r2_mean: R² score
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict] = []

    for name, model in models.items():
        fold_mae = []
        fold_rmse = []
        fold_r2 = []
        
        # Iterate through K folds
        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Train and predict
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)

            # Compute metrics for this fold
            fold_mae.append(mean_absolute_error(y_va, pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_va, pred)))
            fold_r2.append(r2_score(y_va, pred))

        # Store aggregated metrics
        rows.append(
            {
                "model": name,
                "mae_mean": float(np.mean(fold_mae)),
                "mae_std": float(np.std(fold_mae)),
                "rmse_mean": float(np.mean(fold_rmse)),
                "r2_mean": float(np.mean(fold_r2)),
            }
        )

    # Return sorted by MAE (ascending = best models first)
    return pd.DataFrame(rows).sort_values("mae_mean", ascending=True).reset_index(drop=True)
