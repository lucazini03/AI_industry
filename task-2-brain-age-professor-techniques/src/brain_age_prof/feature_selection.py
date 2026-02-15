"""
Feature importance and selection using multiple techniques.

Inspired by: 06-at (Additive Feature Attribution)

Implements three complementary importance measures:
  1. Model Importance (native feature importances from Random Forest)
  2. Permutation Importance: measures drop in performance when feature is shuffled
  3. SHAP Importance: uses Shapley values for additive feature attribution
  4. Boruta: iterative algorithm to identify all relevant features

Final selection is the union of top-K features from each method, ensuring
a diverse set of important features are retained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def model_importance(model: RandomForestRegressor, feature_names: list[str]) -> pd.Series:
    """
    Extract native feature importances from a trained Random Forest.
    
    These are based on impurity decrease (Gini or MSE) across all trees.
    
    Args:
        model (RandomForestRegressor): Fitted Random Forest regressor.
        feature_names (list[str]): Names of features in the model.
    
    Returns:
        pd.Series: Feature importances sorted descending.
    """
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False)


def permutation_importance_series(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute permutation importance: measures drop in model performance when
    a feature is randomly shuffled on a held-out validation set.
    
    Interpretation: A large drop in performance when a feature is shuffled
    indicates that feature is important to the model's predictions.
    
    Advantages over model importance:
      - Model-agnostic (works with any predictor)
      - Captures feature interactions
      - Reflects actual prediction impact
    
    Args:
        model (RandomForestRegressor): Fitted model.
        X (pd.DataFrame): Validation features.
        y (pd.Series): Validation targets.
        n_repeats (int): Number of permutations per feature.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.Series: Permutation importances sorted descending.
    """
    p = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    s = pd.Series(p.importances_mean, index=X.columns)
    return s.sort_values(ascending=False)


def shap_importance_series(model: RandomForestRegressor, X: pd.DataFrame) -> pd.Series:
    """
    Compute SHAP (SHapley Additive exPlanations) feature importance.
    
    SHAP values represent each feature's contribution to moving predictions
    away from the baseline (mean prediction). Based on game-theoretic Shapley values.
    
    Advantages:
      - Theoretically principled (additive feature attribution)
      - Captures global and local feature importance
      - Handles feature interactions
      - More stable across data samples than other measures
    
    Args:
        model (RandomForestRegressor): Fitted model.
        X (pd.DataFrame): Sample of data for SHAP computation (typically validation set).
    
    Returns:
        pd.Series: Mean absolute SHAP values per feature, sorted descending.
    """
    import shap

    # Compute SHAP values using TreeExplainer (fast for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different return formats (list for multi-class, array for regression)
    if isinstance(shap_values, list):
        vals = np.array(shap_values[0])
    else:
        vals = np.array(shap_values)

    # Handle 3D arrays (multi-class case)
    if vals.ndim == 3:
        vals = vals[:, :, 0]

    # Compute mean absolute SHAP value per feature
    mean_abs = np.abs(vals).mean(axis=0)
    s = pd.Series(mean_abs, index=X.columns)
    return s.sort_values(ascending=False)


def boruta_support(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    max_iter: int = 40,
) -> pd.Series:
    """
    Run the Boruta algorithm to identify all relevant features.
    
    Boruta iteratively:
      1. Creates copies of each feature with shuffled values (shadow features)
      2. Trains a classifier/regressor
      3. Compares importance of real vs. shadow features
      4. Removes features performing worse than shadows
      5. Repeats until no more features can be rejected or max iterations reached
    
    Result: A binary series indicating which features are "supported" (important).
    
    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training targets.
        random_state (int): Seed for reproducibility.
        max_iter (int): Maximum iterations.
    
    Returns:
        pd.Series: Boolean series (True = feature is relevant, False = not important).
    """
    # Use a Random Forest as the base estimator for Boruta
    est = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    
    # Run Boruta
    boruta = BorutaPy(
        estimator=est,
        n_estimators="auto",
        random_state=random_state,
        max_iter=max_iter,
        verbose=0,
    )
    boruta.fit(X.values, y.values)
    
    return pd.Series(boruta.support_, index=X.columns)


def build_selected_features(
    perm_imp: pd.Series,
    shap_imp: pd.Series,
    boruta_mask: pd.Series,
    top_k_perm: int,
    top_k_shap: int,
) -> list[str]:
    """
    Combine multiple feature importance measures into a final selection.
    
    Strategy: Take the union of:
      - Top K permutation importance features
      - Top K SHAP importance features
      - All Boruta-supported features
    
    This ensemble approach ensures:
      - Diverse selection across different importance measures
      - Avoids over-reliance on any single method
      - Captures both global and local feature importance
    
    Args:
        perm_imp (pd.Series): Permutation importance scores.
        shap_imp (pd.Series): SHAP importance scores.
        boruta_mask (pd.Series): Boruta boolean support vector.
        top_k_perm (int): Number of top permutation features to keep.
        top_k_shap (int): Number of top SHAP features to keep.
    
    Returns:
        list[str]: Sorted list of selected feature names.
    """
    # Top features from each method
    top_perm = set(perm_imp.head(top_k_perm).index)
    top_shap = set(shap_imp.head(top_k_shap).index)
    boruta_feats = set(boruta_mask[boruta_mask].index)

    # Union of all methods
    selected = sorted((top_perm | top_shap | boruta_feats))
    
    # Fallback if union is empty (rare edge case)
    if not selected:
        selected = perm_imp.head(max(20, top_k_perm // 2)).index.tolist()
    
    return selected
