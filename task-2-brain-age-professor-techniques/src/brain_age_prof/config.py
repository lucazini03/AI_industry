"""
Configuration dataclasses for the brain age prediction pipeline.

These dataclasses encapsulate all hyperparameters and settings for:
  - Imputation: Gaussian Process feature imputation
  - Feature Selection: Permutation, SHAP, and Boruta-based feature importance
  - Training: Model building and cross-validation
"""
from dataclasses import dataclass


@dataclass
class ImputationConfig:
    """
    Configuration for Gaussian Process-based feature imputation.
    
    Attributes:
        max_gp_features (int): Maximum number of features to build GPs for.
            Features with missing values are selected by frequency of missingness.
        n_aux_features (int): Number of auxiliary features (predictors) to use
            in each Gaussian Process regressor. Selected by correlation.
        min_observed (int): Minimum number of non-missing observations required
            to train a GP for a feature. Features below this threshold are skipped.
    """
    max_gp_features: int = 80
    n_aux_features: int = 20
    min_observed: int = 120


@dataclass
class SelectionConfig:
    """
    Configuration for feature selection via importance measures.
    
    Attributes:
        top_k_perm (int): Number of top features to keep from permutation importance.
        top_k_shap (int): Number of top features to keep from SHAP importance.
        boruta_max_iter (int): Maximum iterations for the Boruta algorithm.
    """
    top_k_perm: int = 80
    top_k_shap: int = 80
    boruta_max_iter: int = 40


@dataclass
class TrainingConfig:
    """
    Configuration for model training and validation.
    
    Attributes:
        cv_splits (int): Number of K-fold cross-validation splits.
        random_state (int): Random seed for reproducibility across all components.
        quick (bool): If True, uses lighter models and fewer iterations for faster experiments.
    """
    cv_splits: int = 5
    random_state: int = 42
    quick: bool = False
