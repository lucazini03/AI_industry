"""
Gaussian Process-based feature imputation for missing values.

Inspired by: 03-sf-gp (Filling Missing Values in Traffic Data)

Strategy:
  1. Compute median of all features and use as baseline imputation
  2. For features with significant missingness, train feature-specific Gaussian Process regressors
  3. Each GP uses the top K correlated features as predictors
  4. Predict missing values with trained GPs
  
This approach is more sophisticated than simple mean/median and captures relationships
between features to produce more realistic imputations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class GPColumnModel:
    """
    A trained Gaussian Process model for imputing a specific feature.
    
    Attributes:
        target_col (str): Name of the feature being imputed.
        predictor_cols (list[str]): Names of correlated features used as predictors.
        model (Pipeline): Fitted pipeline with scaler and GP regressor.
    """
    target_col: str
    predictor_cols: list[str]
    model: Pipeline


class GaussianProcessFeatureImputer:
    """
    Feature-wise Gaussian Process imputer for high-dimensional tabular data with missing values.
    
    For each feature with missing values:
      1. Identify the top K most correlated features (computed on observed samples)
      2. Train a GP regressor with those features as inputs
      3. Predict missing values using the trained GP
      4. Features below min_observed threshold or with no missing values are skipped
    
    Example:
        >>> imputer = GaussianProcessFeatureImputer(max_gp_features=80, n_aux_features=20)
        >>> imputer.fit(X_train)
        >>> X_train_imputed = imputer.transform(X_train)
        >>> X_test_imputed = imputer.transform(X_test)
    """

    def __init__(
        self,
        max_gp_features: int = 80,
        n_aux_features: int = 20,
        min_observed: int = 120,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the GP imputer.
        
        Args:
            max_gp_features (int): Max number of features to train GPs for (those with most missingness).
            n_aux_features (int): Number of correlated predictors per GP.
            min_observed (int): Minimum non-missing observations to train a GP.
            random_state (int): Seed for reproducibility.
        """
        self.max_gp_features = max_gp_features
        self.n_aux_features = n_aux_features
        self.min_observed = min_observed
        self.random_state = random_state

        self.medians_: pd.Series | None = None
        self.models_: list[GPColumnModel] = []
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "GaussianProcessFeatureImputer":
        """
        Fit the imputer by training GP models for features with missing values.
        
        Args:
            X (pd.DataFrame): Training data with potential missing values (NaN).
        
        Returns:
            self: For chaining.
        """
        X = X.copy()
        self.columns_ = list(X.columns)
        
        # Baseline: compute medians for fallback imputation
        self.medians_ = X.median(numeric_only=True)
        base = X.fillna(self.medians_)

        # Identify features with missing values, sorted by missingness frequency
        missing_ratio = X.isna().mean().sort_values(ascending=False)
        candidate_cols = [c for c, v in missing_ratio.items() if v > 0][: self.max_gp_features]

        self.models_ = []
        for col in candidate_cols:
            observed = X[col].notna()
            
            # Skip if not enough observed values to train GP
            if observed.sum() < self.min_observed:
                continue

            # Find top K correlated features (among observed samples)
            corr = base.loc[observed].corrwith(base.loc[observed, col]).abs().drop(labels=[col], errors="ignore")
            predictor_cols = corr.nlargest(self.n_aux_features).index.tolist()
            
            if len(predictor_cols) == 0:
                continue

            # Build and fit pipeline: StandardScaler -> GaussianProcessRegressor
            kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-3,
                normalize_y=True,
                random_state=self.random_state,
                n_restarts_optimizer=0,
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("gp", gp),
            ])
            pipe.fit(base.loc[observed, predictor_cols], base.loc[observed, col])

            self.models_.append(GPColumnModel(col, predictor_cols, pipe))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation to data.
        
        Args:
            X (pd.DataFrame): Data with potential missing values.
        
        Returns:
            pd.DataFrame: Data with missing values imputed (or filled with median if no GP trained).
        
        Raises:
            RuntimeError: If called before fit().
        """
        if self.medians_ is None:
            raise RuntimeError("Imputer must be fitted before transform().")

        X = X.copy()
        original_missing = X.isna()
        
        # Start with median fill
        out = X.fillna(self.medians_)

        # For each feature with a trained GP, replace missing values with GP predictions
        for col_model in self.models_:
            miss_mask = original_missing[col_model.target_col]
            if miss_mask.sum() == 0:
                continue
            pred = col_model.model.predict(out.loc[miss_mask, col_model.predictor_cols])
            out.loc[miss_mask, col_model.target_col] = pred

        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform data in one step.
        
        Args:
            X (pd.DataFrame): Training data with missing values.
        
        Returns:
            pd.DataFrame: Imputed data.
        """
        return self.fit(X).transform(X)
