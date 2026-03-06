"""
functions.py  –  Property Sales Forecasting
============================================

Centralised library of every reusable building-block consumed by the analysis
notebook (sales_updated.ipynb).

Keeping logic here rather than inline in notebook cells:
  * removes code duplication across cells
  * makes unit-testing and re-use straightforward
  * keeps the notebook focused on narrative and results

Module layout
-------------
1.  Imports & globals
2.  Data cleaning / imputation          → imputer_pipeline
3.  Outlier strategies & aggregation    → pipeline  (+ private plot helpers)
4.  Feature engineering                 → add_seasonality_features, add_price_lags
5.  Feature catalogue                   → DEFAULT_FEATURES, FEATURE_SETS,
                                           DEFAULT_LSTM_PARAM_GRID
6.  Metrics                             → calculate_rmse
7.  Visualisation helpers               → visualize_distribution,
                                           plot_price_distribution,
                                           plot_price_trend,
                                           plot_monthly_trend,
                                           plot_correlation
8.  Random-Forest pipeline              → random_forest_evaluation_pipeline
9.  LSTM helpers & pipelines            → create_sequences,
                                           run_lstm_univariate,
                                           run_lstm_univariate_seq2seq,
                                           run_lstm_multivariate2,
                                           gridsearch_lstm,
                                           lstm_evaluation_pipeline_univariate,
                                           lstm_evaluation_pipeline2
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data cleaning / imputation
# ─────────────────────────────────────────────────────────────────────────────

def imputer_pipeline(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing / zero bedroom counts using K-Nearest Neighbours (KNN).

    Zero-bedroom entries are treated as recording artefacts and replaced with
    ``NaN`` before imputation.  KNN is preferred over simple median imputation
    because it exploits the correlations between the number of bedrooms,
    property type, postcode, and price.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame.  Expected columns:
        ``date``, ``type``, ``price``, ``bedrooms``, ``postcode``.
    n_neighbors : int, optional
        Number of nearest neighbours used by
        :class:`sklearn.impute.KNNImputer`.  Default: ``5``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``bedrooms`` imputed and rounded to the nearest
        non-negative integer.

    Notes
    -----
    * All feature columns are z-score normalised before imputation so that
      features with large absolute values (price, postcode) do not dominate
      the KNN distance metric.
    * The scaler and encoder are fit on the **entire** DataFrame — this is
      acceptable because imputation is performed before any train/test split
      and ``bedrooms`` is not the forecasting target.
    """
    df = df.copy()

    # ── Replace 0 bedrooms with NaN (treat as missing) ───────────────────────
    df["bedrooms"] = df["bedrooms"].replace(0, np.nan)

    # ── Encode the categorical 'type' column numerically ─────────────────────
    le = LabelEncoder()
    df["_type_enc"] = le.fit_transform(df["type"])

    # ── Build the feature matrix that drives KNN distances ───────────────────
    impute_cols = ["bedrooms", "price", "postcode", "_type_enc"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[impute_cols])

    # ── Run KNN imputer ───────────────────────────────────────────────────────
    imputed = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)

    # ── Inverse-transform and write back only the bedroom column ─────────────
    restored = scaler.inverse_transform(imputed)
    df["bedrooms"] = np.maximum(0, np.round(restored[:, 0])).astype(int)

    # ── Remove temporary helper column ───────────────────────────────────────
    df = df.drop(columns=["_type_enc"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Outlier strategies & monthly aggregation
# ─────────────────────────────────────────────────────────────────────────────

# ── Private helpers called internally by `pipeline` ──────────────────────────

def _plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of the ``price`` column (private pipeline helper)."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], kde=True)
    plt.title("Distribution of Property Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def _plot_price_trend(monthly: pd.DataFrame) -> None:
    """Line chart of monthly median price with a 3-month rolling average
    (private pipeline helper)."""
    plt.figure(figsize=(12, 6))
    plt.plot(
        monthly.index, monthly["price"],
        color="lightgray", label="Original", linewidth=1.5,
    )
    plt.plot(
        monthly.index, monthly["price_rolling"],
        color="blue", label="Smoothed (3-month)", linewidth=2,
    )
    plt.title("Median Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Median Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_monthly_trend(monthly_df: pd.DataFrame) -> None:
    """Four-panel subplot of price, avg_bedrooms, house_ratio, sales_count
    (private pipeline helper)."""
    monthly_df[["price", "avg_bedrooms", "house_ratio", "sales_count"]].plot(
        subplots=True, figsize=(10, 8), title="Monthly Trends"
    )
    plt.tight_layout()
    plt.show()


def pipeline(
    df: pd.DataFrame,
    type_of_outlier: str,
    plotting: bool = True,
) -> tuple:
    """Clean the raw sales data and aggregate it to monthly frequency.

    Applies one of four price-outlier strategies, then computes monthly
    median price and a set of market-structure features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame with columns
        ``date``, ``price``, ``type``, ``bedrooms``, ``postcode``.
    type_of_outlier : {'none', 'remove', 'log_transform', 'power_transform'}
        Strategy for handling right-skewed price outliers:

        * ``'none'``            – leave prices unchanged.
        * ``'remove'``          – drop rows outside the IQR fences
          (Q1 − 1.5 × IQR, Q3 + 1.5 × IQR).
        * ``'log_transform'``   – apply a natural-log transform to ``price``.
          Remember to exponentiate predictions before comparing to dollar
          values.
        * ``'power_transform'`` – apply a Box-Cox transform via
          :class:`sklearn.preprocessing.PowerTransformer`.

    plotting : bool, optional
        If ``True`` (default), renders three diagnostic plots:
        the price distribution histogram, the rolling median trend, and
        four-panel monthly trends.

    Returns
    -------
    df_cleaned : pd.DataFrame
        Row-level data after applying the chosen outlier strategy.
    monthly : pd.DataFrame
        Monthly median price time-series with a ``price_rolling`` column
        (3-month centred moving average).  Index is a ``DatetimeIndex``
        at month-end frequency.
    monthly_df : pd.DataFrame
        Monthly aggregates indexed by month-start timestamp, containing:
        ``price`` (median), ``house_ratio`` (fraction of 'house' type),
        ``avg_bedrooms``, ``sales_count``.

    Raises
    ------
    ValueError
        If *type_of_outlier* is not one of the four recognised strategies.

    Notes
    -----
    ``house_ratio`` captures the mix of property types each month.  A month
    dominated by houses will naturally show a higher median price than one
    dominated by units, so including this ratio helps downstream models
    disentangle type-composition effects from genuine price trends.
    """
    # ── Apply outlier strategy ────────────────────────────────────────────────
    if type_of_outlier == "none":
        df_cleaned = df.copy()

    elif type_of_outlier == "remove":
        Q1 = df["price"].quantile(0.25)
        Q3 = df["price"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df[
            (df["price"] >= lower_bound) & (df["price"] <= upper_bound)
        ].copy()

    elif type_of_outlier == "log_transform":
        df_cleaned = df.copy()
        df_cleaned["price"] = np.log(df_cleaned["price"])

    elif type_of_outlier == "power_transform":
        df_cleaned = df.copy()
        pt = PowerTransformer(method="box-cox")
        df_cleaned["price"] = pt.fit_transform(df_cleaned[["price"]])

    else:
        raise ValueError(
            f"Unknown outlier strategy '{type_of_outlier}'. "
            "Choose from: 'none', 'remove', 'log_transform', 'power_transform'."
        )

    # ── Optional diagnostic plot ──────────────────────────────────────────────
    if plotting:
        _plot_price_distribution(df_cleaned)

    # ── Monthly median + rolling average ─────────────────────────────────────
    monthly = df_cleaned.resample("ME", on="date").agg({"price": "median"})
    monthly["price_rolling"] = monthly["price"].rolling(window=3, center=True).mean()

    if plotting:
        _plot_price_trend(monthly)

    # ── Monthly feature aggregation ───────────────────────────────────────────
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"])
    df_cleaned["month"] = df_cleaned["date"].dt.to_period("M")

    monthly_df = (
        df_cleaned.groupby("month")
        .agg(
            price=("price", "median"),
            house_ratio=("type", lambda x: (x == "house").mean()),
            avg_bedrooms=("bedrooms", "mean"),
            sales_count=("price", "count"),
        )
        .reset_index()
    )

    monthly_df["month"] = monthly_df["month"].dt.to_timestamp()
    monthly_df.set_index("month", inplace=True)

    if plotting:
        _plot_monthly_trend(monthly_df)

    return df_cleaned, monthly, monthly_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_seasonality_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Append cyclical month encoding and a scaled year to a monthly DataFrame.

    Cyclical encoding maps the integer month (1–12) onto the unit circle so
    that the model "knows" December and January are adjacent.  A raw integer
    feature would imply they are 11 steps apart, which is incorrect.

    Columns added
    -------------
    ``month_sin``
        ``sin(2π × month / 12)``
    ``month_cos``
        ``cos(2π × month / 12)``
    ``year_scaled``
        Year standardised to zero mean and unit variance.  Acts as a linear
        proxy for the long-run price trend without dominating gradient-based
        optimisers — this is especially important for stable LSTM training.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` at monthly frequency.

    Returns
    -------
    pd.DataFrame
        Copy of *monthly_df* with three additional columns appended.

    Example
    -------
    >>> monthly_df = add_seasonality_features(monthly_df)
    >>> monthly_df[["month_sin", "month_cos", "year_scaled"]].head()
    """
    df = monthly_df.copy()
    month = df.index.month
    year = np.asarray(df.index.year, dtype=float)

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    year_mean = year.mean()
    year_std = year.std() if year.std() != 0 else 1.0
    df["year_scaled"] = (year - year_mean) / year_std

    return df


def add_price_lags(df: pd.DataFrame, lags: tuple = (1, 12)) -> pd.DataFrame:
    """Create lagged price features for autoregressive modelling.

    Each ``price_lag_k`` column at row *t* equals ``price`` at row *t − k*.
    Rows 0 … (max_lag − 1) will contain ``NaN`` and must be dropped before
    model training (use ``df.dropna()``).

    Parameters
    ----------
    df : pd.DataFrame
        Monthly DataFrame that must contain a ``price`` column.
    lags : tuple of int, optional
        Lag offsets to create.  Default: ``(1, 12)`` — one month back and
        one year back.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``price_lag_<k>`` columns appended for each *k*
        in *lags*.

    Example
    -------
    >>> df = add_price_lags(monthly_df, lags=(1, 3, 12))
    >>> df[["price", "price_lag_1", "price_lag_3", "price_lag_12"]].head()
    """
    df = df.copy()
    for lag in lags:
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Feature catalogue
# ─────────────────────────────────────────────────────────────────────────────

#: Feature sets used by the Random-Forest evaluation pipeline.
#:
#: ``known_only``
#:     Market features that are known at forecast time (bedrooms mix,
#:     house/unit ratio, transaction volume).
#: ``seasonal``
#:     Same as *known_only*, plus cyclical month encoding and a year trend.
#: ``seasonal_plus_lags``
#:     All of *seasonal*, plus autoregressive price lags.
#:     Requires :func:`add_price_lags` to have been called first.
DEFAULT_FEATURES: dict = {
    "known_only": [
        "avg_bedrooms",
        "house_ratio",
        "sales_count",
    ],
    "seasonal": [
        "month_sin",
        "month_cos",
        "year_scaled",
        "avg_bedrooms",
        "house_ratio",
        "sales_count",
    ],
    "seasonal_plus_lags": [
        "month_sin",
        "month_cos",
        "year_scaled",
        "avg_bedrooms",
        "house_ratio",
        "sales_count",
        "price_lag_1",
        "price_lag_12",
    ],
}

#: Feature sets used by the LSTM evaluation pipelines.
#:
#: ``known_future``
#:     Only deterministic calendar features — safe for any forecast horizon
#:     because sin/cos/year values are always known in advance.
#: ``market_unknown``
#:     Market-structure features that may not be available in a true
#:     out-of-sample scenario (e.g. avg bedrooms and sales count require
#:     knowing who will transact next month).
#: ``all``
#:     Union of *known_future* and *market_unknown*.
FEATURE_SETS: dict = {
    "known_future": [
        "month_sin",
        "month_cos",
        "year_scaled",
    ],
    "market_unknown": [
        "avg_bedrooms",
        "house_ratio",
        "sales_count",
    ],
    "all": [
        "month_sin",
        "month_cos",
        "year_scaled",
        "avg_bedrooms",
        "house_ratio",
        "sales_count",
    ],
}

#: Default hyperparameter grid passed to :func:`gridsearch_lstm`.
DEFAULT_LSTM_PARAM_GRID: dict = {
    "lstm_units":  [32, 64],
    "dense_units": [16, 32],
    "epochs":      [30, 50],
}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE in the same units as *y_true* / *y_pred*.

    Notes
    -----
    When prices have been log-transformed, pass the back-transformed (dollar)
    arrays so that the returned RMSE is expressed in dollars::

        rmse = calculate_rmse(np.exp(y_test), np.exp(y_pred))
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def visualize_distribution(df: pd.DataFrame, column: str) -> None:
    """Side-by-side histogram and box-plot for a single numerical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains *column*.
    column : str
        Name of the column to visualise.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[column], kde=True, ax=ax[0])
    ax[0].set_title("Histogram")

    sns.boxplot(x=df[column], ax=ax[1])
    ax[1].set_title("Box Plot")

    plt.suptitle(f"Distribution of '{column}'", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram with KDE overlay for the ``price`` column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains a ``price`` column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df["price"], kde=True)
    plt.title("Distribution of Property Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_price_trend(monthly: pd.DataFrame) -> None:
    """Line chart of monthly median price with a 3-month rolling average.

    Parameters
    ----------
    monthly : pd.DataFrame
        Monthly aggregated DataFrame with columns ``price`` and
        ``price_rolling``.  Typically the second return value of
        :func:`pipeline`.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        monthly.index, monthly["price"],
        color="lightgray", label="Original", linewidth=1.5,
    )
    plt.plot(
        monthly.index, monthly["price_rolling"],
        color="blue", label="Smoothed (3-month)", linewidth=2,
    )
    plt.title("Median Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Median Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_monthly_trend(monthly_df: pd.DataFrame) -> None:
    """Four-panel subplot: price, avg_bedrooms, house_ratio, sales_count.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly aggregated DataFrame.  Typically the third return value of
        :func:`pipeline`.
    """
    monthly_df[["price", "avg_bedrooms", "house_ratio", "sales_count"]].plot(
        subplots=True, figsize=(10, 8), title="Monthly Trends"
    )
    plt.tight_layout()
    plt.show()


def plot_correlation(
    df: pd.DataFrame,
    feature: str,
    target: str = "price",
) -> None:
    """Scatter plot of *feature* vs *target* with the figure saved to disk.

    The figure is saved as ``correlation_<feature>_<target>.png`` in the
    current working directory **before** ``plt.show()`` is called, which
    prevents the file from being written as an empty image.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both *feature* and *target* columns.
    feature : str
        Column name for the x-axis.
    target : str, optional
        Column name for the y-axis.  Default: ``'price'``.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[feature], y=df[target], alpha=0.6)
    plt.title(f"Correlation: {feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"correlation_{feature}_{target}.png", dpi=150)  # save BEFORE show
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Random-Forest evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def random_forest_evaluation_pipeline(
    df: pd.DataFrame,
    pipeline_function,
    add_seasonality_function,
    train_size: float = 0.8,
    feature_dict: dict = None,
    feature_keys: tuple = ("known_only", "seasonal"),
    outlier_methods: tuple = ("none",),
    use_lags: bool = True,
    lag_values: tuple = (1, 12),
    plot: bool = True,
    use_shap: bool = False,
    grid_search: bool = False,
    rf_params: dict = None,
) -> None:
    """Evaluate a Random-Forest regressor across outlier strategies and feature sets.

    This fully-parametric pipeline:

    1. Applies the chosen *outlier_methods* via *pipeline_function*.
    2. Optionally adds seasonality features and autoregressive price lags.
    3. Performs a **chronological** train/test split (no shuffling).
    4. Optionally runs :class:`~sklearn.model_selection.GridSearchCV` for
       hyper-parameter tuning.
    5. Reports RMSE in dollar terms (back-transforms log prices automatically).
    6. Optionally renders forecast plots, feature-importance bar charts, and
       SHAP explanations.

    Parameters
    ----------
    df : pd.DataFrame
        Original raw sales DataFrame.
    pipeline_function : callable
        Data-cleaning pipeline with signature::

            pipeline(df, type_of_outlier, plotting) → (df_cleaned, monthly, monthly_df)

        Use :func:`pipeline` from this module.
    add_seasonality_function : callable
        Feature-engineering function with signature::

            add_seasonality_features(monthly_df) → monthly_df

        Use :func:`add_seasonality_features` from this module.
    train_size : float, optional
        Fraction of the time-ordered data used for training.  Default: ``0.8``.
    feature_dict : dict, optional
        Mapping of feature-set name → list of column names.  Defaults to
        :data:`DEFAULT_FEATURES`.
    feature_keys : tuple of str, optional
        Which keys in *feature_dict* to evaluate.
        Default: ``('known_only', 'seasonal')``.
    outlier_methods : tuple of str, optional
        Outlier strategies to iterate over (must be valid values for
        *pipeline_function*).  Default: ``('none',)``.
    use_lags : bool, optional
        If ``True``, append lagged price columns via :func:`add_price_lags`.
        Required when *feature_keys* includes ``'seasonal_plus_lags'``.
        Default: ``True``.
    lag_values : tuple of int, optional
        Lag offsets to create when *use_lags* is ``True``.  Default: ``(1, 12)``.
    plot : bool, optional
        If ``True``, render forecast plots and feature-importance charts.
        Default: ``True``.
    use_shap : bool, optional
        If ``True``, compute and display SHAP summary plots (can be slow for
        large test sets).  Default: ``False``.
    grid_search : bool, optional
        If ``True``, run :class:`~sklearn.model_selection.GridSearchCV` to
        find the best hyper-parameters before fitting the final model.
        Default: ``False``.
    rf_params : dict, optional
        Keyword arguments forwarded to
        :class:`~sklearn.ensemble.RandomForestRegressor` when *grid_search* is
        ``False``.  Defaults to
        ``{'n_estimators': 300, 'max_depth': None, 'random_state': 42, 'n_jobs': -1}``.
    """
    if feature_dict is None:
        feature_dict = DEFAULT_FEATURES

    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
        }

    _grid_params = {
        "n_estimators": range(100, 501, 100),
        "max_depth": range(5, 16, 5),
        "min_samples_split": [2, 3, 4, 5],
    }

    for outlier in outlier_methods:
        for key in feature_keys:
            print("\n" + "=" * 60)
            print(f"RF | Outlier: {outlier} | Features: {key}")
            print("=" * 60)

            # ── 1. Build pipeline data ────────────────────────────────────────
            _, _, monthly_df = pipeline_function(
                df, type_of_outlier=outlier, plotting=False
            )
            monthly_df = add_seasonality_function(monthly_df)

            if use_lags:
                monthly_df = add_price_lags(monthly_df, lag_values)

            monthly_df = monthly_df.dropna()
            feature_cols = feature_dict[key]

            # ── 2. Chronological train/test split ─────────────────────────────
            split_idx = int(len(monthly_df) * train_size)
            train = monthly_df.iloc[:split_idx]
            test  = monthly_df.iloc[split_idx:]

            X_train, y_train = train[feature_cols], train["price"]
            X_test,  y_test  = test[feature_cols],  test["price"]

            # ── 3. Fit model ──────────────────────────────────────────────────
            if grid_search:
                print("  Running GridSearchCV …")
                gs = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    _grid_params, cv=3, n_jobs=-1,
                )
                gs.fit(X_train, y_train)
                model_rf = gs.best_estimator_
                print(f"  Best params: {gs.best_params_}")
            else:
                model_rf = RandomForestRegressor(**rf_params)
                model_rf.fit(X_train, y_train)

            preds = model_rf.predict(X_test)

            # ── 4. Back-transform log prices to dollars ───────────────────────
            if outlier == "log_transform":
                actual  = np.exp(y_test.values)
                preds   = np.exp(preds)
                history = np.exp(monthly_df["price"].values)
            else:
                actual  = y_test.values
                history = monthly_df["price"].values

            # ── 5. RMSE ───────────────────────────────────────────────────────
            rmse = calculate_rmse(actual, preds)
            print(f"  RMSE = ${rmse:,.2f}")

            if not plot:
                continue

            # ── 6. Forecast plot ──────────────────────────────────────────────
            plt.figure(figsize=(12, 6))
            plt.plot(monthly_df.index, history, color="lightgray", label="History")
            plt.plot(test.index, actual, color="black", label="Actual")
            plt.plot(
                test.index, preds, "--",
                label=f"RF (RMSE={int(rmse):,})",
            )
            plt.title(f"RF Forecast  |  features: {key}  |  outlier: {outlier}")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # ── 7. Feature importance ─────────────────────────────────────────
            importance = (
                pd.Series(model_rf.feature_importances_, index=feature_cols)
                .sort_values(ascending=False)
            )
            print("\n  Feature Importance:")
            print(importance.round(4).to_string())

            plt.figure(figsize=(8, 4))
            sns.barplot(x=importance.values, y=importance.index, palette="viridis")
            plt.title(f"RF Feature Importance  |  {key}  |  {outlier}")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.show()

            # ── 8. SHAP (optional) ────────────────────────────────────────────
            if use_shap:
                explainer  = shap.TreeExplainer(model_rf)
                shap_values = explainer.shap_values(X_test)

                plt.figure()
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.title(f"SHAP Mean |value|  |  {key}  |  {outlier}")
                plt.tight_layout()
                plt.show()

                plt.figure()
                shap.summary_plot(shap_values, X_test, show=False)
                plt.title(f"SHAP Beeswarm  |  {key}  |  {outlier}")
                plt.tight_layout()
                plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  LSTM helpers & pipelines
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 12,
) -> tuple:
    """Convert a scaled time-series array into (input window, target) pairs.

    Generates overlapping sliding windows of length *seq_len* over *X*,
    each paired with the *y* value immediately following the window.

    Parameters
    ----------
    X : np.ndarray, shape (n_timesteps, n_features)
        Feature matrix.  Each row is one timestep.
    y : np.ndarray, shape (n_timesteps, 1) or (n_timesteps,)
        Target vector aligned with *X*.
    seq_len : int, optional
        Length of each input window.  Default: ``12`` (one calendar year).

    Returns
    -------
    Xs : np.ndarray, shape (n_samples, seq_len, n_features)
        Array of input windows.
    ys : np.ndarray
        Corresponding target values.

    Notes
    -----
    The number of generated samples is ``len(X) - seq_len``.  The first
    *seq_len* rows of *X* are used only as context and never appear as targets.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def run_lstm_univariate(
    monthly_df_cleaned: pd.DataFrame,
    train_size: float,
    seq_len: int = 12,
    lstm_units: int = 64,
    dense_units: int = 32,
    epochs: int = 40,
    batch_size: int = 16,
) -> tuple:
    """Train a univariate recursive LSTM on monthly price data.

    The model predicts one step at a time using only past price values.
    During evaluation, each new prediction is fed back as input for the next
    step — the model **never** sees the true future price.

    Architecture::

        LSTM(lstm_units) → Dense(dense_units, relu) → Dense(1)

    Note: in a single forward pass this architecture outputs **one** price
    prediction.  The full test sequence is produced by calling the model
    once per test step (recursive / autoregressive forecasting).

    Parameters
    ----------
    monthly_df_cleaned : pd.DataFrame
        Monthly DataFrame with a ``price`` column.
    train_size : float
        Fraction of data used for training (chronological split).
    seq_len : int, optional
        Look-back window in months.  Default: ``12``.
    lstm_units : int, optional
        Number of LSTM hidden units.  Default: ``64``.
    dense_units : int, optional
        Hidden size of the intermediate Dense layer.  Default: ``32``.
    epochs : int, optional
        Training epochs.  Default: ``40``.
    batch_size : int, optional
        Mini-batch size.  Default: ``16``.

    Returns
    -------
    preds : np.ndarray, shape (n_test, 1)
        Predicted prices in original (inverse-scaled) units.
    y_test_real : np.ndarray, shape (n_test, 1)
        Ground-truth test prices.
    mse : float
        Mean Squared Error on the test set (in scaled units).
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    split_idx = int(len(monthly_df_cleaned) * train_size)
    train_df = monthly_df_cleaned.iloc[:split_idx]
    test_df  = monthly_df_cleaned.iloc[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[["price"]])
    test_scaled  = scaler.transform(test_df[["price"]])

    X_train, y_train = create_sequences(train_scaled, train_scaled, seq_len)

    model = Sequential([
        LSTM(lstm_units, input_shape=(seq_len, 1)),
        Dense(dense_units, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # ── Recursive forecasting: feed own predictions back as input ─────────────
    current_seq  = train_scaled[-seq_len:]
    preds_scaled = []
    for _ in range(len(test_scaled)):
        next_pred = model.predict(
            current_seq.reshape(1, seq_len, 1), verbose=0
        )[0][0]
        preds_scaled.append(next_pred)
        current_seq = np.append(current_seq[1:], [[next_pred]], axis=0)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds        = scaler.inverse_transform(preds_scaled)
    y_test_real  = test_df[["price"]].values
    mse          = mean_squared_error(y_test_real, preds)

    return preds, y_test_real, mse


def run_lstm_univariate_seq2seq(
    monthly_df_cleaned: pd.DataFrame,
    train_size: float,
    seq_len: int = 12,
    horizon: int = 12,
    lstm_units: int = 64,
    dense_units: int = 32,
    epochs: int = 40,
    batch_size: int = 16,
) -> tuple:
    """Train a sequence-to-sequence (multi-step) univariate LSTM.

    Unlike :func:`run_lstm_univariate`, this model predicts the next *horizon*
    prices in a **single forward pass** — no recursion required.  This is
    more suitable when a fixed multi-month forecast is needed and recursive
    error accumulation is a concern.

    Architecture::

        LSTM(lstm_units) → Dense(dense_units, relu) → Dense(horizon)

    Parameters
    ----------
    monthly_df_cleaned : pd.DataFrame
        Monthly DataFrame with a ``price`` column.
    train_size : float
        Fraction of data used for training.
    seq_len : int, optional
        Input look-back window (months).  Default: ``12``.
    horizon : int, optional
        Number of future months to forecast in one shot.  Default: ``12``.
        Must not exceed the number of test observations.
    lstm_units, dense_units, epochs, batch_size : int, optional
        Model hyper-parameters (see :func:`run_lstm_univariate`).

    Returns
    -------
    model : keras.Sequential
        Trained Keras model.
    preds : np.ndarray, shape (horizon,)
        Forecast prices in original (dollar) units.
    actual : np.ndarray, shape (horizon,)
        Ground-truth prices for the forecast window.
    mse : float
        Mean Squared Error over the forecast horizon.
    test_index : pd.DatetimeIndex
        Date index corresponding to the *horizon* forecast steps.

    Raises
    ------
    ValueError
        If *horizon* exceeds the available test-set length.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    prices    = monthly_df_cleaned[["price"]].values
    split_idx = int(len(prices) * train_size)
    train_vals = prices[:split_idx]
    test_vals  = prices[split_idx:]

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    # ── Build multi-step (X, y) pairs ─────────────────────────────────────────
    def _multistep(series, s_len, h):
        Xs, ys = [], []
        for i in range(len(series) - s_len - h + 1):
            Xs.append(series[i : i + s_len])
            ys.append(series[i + s_len : i + s_len + h].flatten())
        return np.array(Xs), np.array(ys)

    X_train, y_train = _multistep(train_scaled, seq_len, horizon)

    model = Sequential([
        LSTM(lstm_units, input_shape=(seq_len, 1)),
        Dense(dense_units, activation="relu"),
        Dense(horizon),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    if len(test_scaled) < horizon:
        raise ValueError(
            f"horizon={horizon} is longer than the test set "
            f"({len(test_scaled)} steps).  "
            "Reduce horizon or increase train_size."
        )

    last_window  = train_scaled[-seq_len:].reshape(1, seq_len, 1)
    preds_scaled = model.predict(last_window, verbose=0)[0].reshape(-1, 1)

    preds  = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(test_scaled[:horizon]).flatten()
    mse    = mean_squared_error(actual, preds)

    test_index = monthly_df_cleaned.iloc[split_idx : split_idx + horizon].index

    return model, preds, actual, mse, test_index


def run_lstm_multivariate2(
    monthly_df: pd.DataFrame,
    feature_cols_exog: list,
    train_size: float = 0.85,
    seq_len: int = 12,
    lstm_units: int = 64,
    dense_units: int = 32,
    epochs: int = 40,
    batch_size: int = 16,
) -> tuple:
    """Train a multivariate autoregressive LSTM.

    The input feature vector at each timestep is::

        [price_lag, exog_feature_1, exog_feature_2, …]

    The past price is always included as the first feature.  During test-set
    forecasting, the model's own prediction at step *t* is substituted for the
    unknown future price at step *t+1* (recursive strategy).

    Key design choices
    ------------------
    * **No data leakage**: scalers are fit exclusively on the training split.
    * **Recursive forecasting**: only exogenous features (e.g. calendar,
      market-structure) are taken from the test set.  The price column is
      always replaced with the model's own previous prediction.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly DataFrame with ``price`` and all columns in *feature_cols_exog*.
    feature_cols_exog : list of str
        Exogenous feature columns appended to the lagged price input.
    train_size : float, optional
        Training fraction.  Default: ``0.85``.
    seq_len : int, optional
        Input look-back window.  Default: ``12``.
    lstm_units, dense_units, epochs, batch_size : int, optional
        Model hyper-parameters.

    Returns
    -------
    model : keras.Sequential
        Trained Keras model.
    preds : np.ndarray, shape (n_test,)
        Forecast prices in original (dollar) units.
    actual : np.ndarray, shape (n_test,)
        Ground-truth test prices.
    rmse : float
        RMSE in original price units.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    df = monthly_df.copy()
    target       = "price"
    feature_cols = [target] + feature_cols_exog   # price is always feature 0

    split_idx = int(len(df) * train_size)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(train_df[feature_cols])
    scaler_y.fit(train_df[[target]])

    train_X = scaler_X.transform(train_df[feature_cols])
    test_X  = scaler_X.transform(test_df[feature_cols])
    train_y = scaler_y.transform(train_df[[target]])
    test_y  = scaler_y.transform(test_df[[target]])

    X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_len)

    model = Sequential([
        LSTM(lstm_units, input_shape=(seq_len, len(feature_cols))),
        Dense(dense_units, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs, batch_size=batch_size, verbose=0,
    )

    # ── Recursive forecasting ─────────────────────────────────────────────────
    current_seq  = train_X[-seq_len:].copy()
    preds_scaled = []

    for i in range(len(test_df)):
        pred = model.predict(
            current_seq.reshape(1, seq_len, len(feature_cols)), verbose=0
        )[0, 0]
        preds_scaled.append(pred)

        next_row    = test_X[i].copy()
        next_row[0] = pred          # substitute unknown future price
        current_seq = np.vstack([current_seq[1:], next_row])

    preds  = scaler_y.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()
    actual = scaler_y.inverse_transform(test_y).flatten()
    rmse   = float(np.sqrt(mean_squared_error(actual, preds)))

    return model, preds, actual, rmse


def gridsearch_lstm(
    monthly_df: pd.DataFrame,
    feature_cols: list,
    param_grid: dict,
    seq_len: int,
    train_size: float,
) -> tuple:
    """Grid search over LSTM hyper-parameters using :func:`run_lstm_multivariate2`.

    Iterates over every combination in *param_grid* (Cartesian product),
    trains a model for each, and records the RMSE.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly DataFrame passed through to :func:`run_lstm_multivariate2`.
    feature_cols : list of str
        Exogenous feature columns (same meaning as *feature_cols_exog* in
        :func:`run_lstm_multivariate2`).
    param_grid : dict
        Hyper-parameter search space.  Keys must be valid keyword arguments
        of :func:`run_lstm_multivariate2`, values are lists of candidates.
        Example::

            {
                "lstm_units":  [32, 64],
                "dense_units": [16, 32],
                "epochs":      [30, 50],
            }

    seq_len : int
        Look-back window forwarded to :func:`run_lstm_multivariate2`.
    train_size : float
        Training fraction forwarded to :func:`run_lstm_multivariate2`.

    Returns
    -------
    best : tuple[dict, float]
        ``(best_params, best_rmse)`` — the configuration that achieved the
        lowest RMSE on the test split.
    results : list of tuple[dict, float]
        All ``(params, rmse)`` pairs evaluated, in iteration order.
    """
    best    = None
    results = []

    for params in product(*param_grid.values()):
        cfg = dict(zip(param_grid.keys(), params))

        _, _, _, rmse = run_lstm_multivariate2(
            monthly_df,
            feature_cols_exog=feature_cols,
            train_size=train_size,
            seq_len=seq_len,
            **cfg,
        )
        results.append((cfg, rmse))
        print(f"  {cfg}  →  RMSE: {rmse:,.2f}")

        if best is None or rmse < best[1]:
            best = (cfg, rmse)

    print(f"\n  Best configuration: {best}")
    return best, results


def lstm_evaluation_pipeline_univariate(
    df: pd.DataFrame,
    pipeline_function,
    types_of_outlier: tuple = ("none", "remove", "log_transform"),
    plot: bool = True,
    models: tuple = ("univariate", "sequential"),
    seq_len: int = 12,
    horizon: int = 12,
    train_size: float = 0.85,
) -> None:
    """Evaluate univariate LSTM variants across multiple outlier strategies.

    Supports two model types:

    * ``'univariate'``  – recursive 1-step LSTM (:func:`run_lstm_univariate`).
    * ``'sequential'``  – seq2seq multi-step LSTM
      (:func:`run_lstm_univariate_seq2seq`).

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame.
    pipeline_function : callable
        Cleaning / aggregation pipeline (see :func:`pipeline`).
    types_of_outlier : tuple of str, optional
        Outlier strategies to evaluate.
    plot : bool, optional
        If ``True``, render forecast comparison plots.  Default: ``True``.
    models : tuple of str, optional
        Which LSTM variants to run.  Any subset of
        ``('univariate', 'sequential')``.
    seq_len : int, optional
        Look-back window for both LSTM types.  Default: ``12``.
    horizon : int, optional
        Forecast horizon for the seq2seq model.  Default: ``12``.
    train_size : float, optional
        Training fraction.  Default: ``0.85``.
    """
    for type_of_outlier in types_of_outlier:
        _, _, monthly_df_cleaned = pipeline_function(
            df, type_of_outlier=type_of_outlier, plotting=False
        )
        print(f"\n=== Univariate LSTM | outlier: {type_of_outlier} ===")

        # ── Recursive 1-step model ────────────────────────────────────────────
        if "univariate" in models:
            preds_uni, y_uni, mse_uni = run_lstm_univariate(
                monthly_df_cleaned, train_size=train_size, seq_len=seq_len
            )
            if type_of_outlier == "log_transform":
                actual_uni  = np.exp(y_uni.flatten())
                preds_uni_d = np.exp(preds_uni.flatten())
            else:
                actual_uni  = y_uni.flatten()
                preds_uni_d = preds_uni.flatten()

            rmse_uni = calculate_rmse(actual_uni, preds_uni_d)
            print(f"  1-step Recursive RMSE = ${rmse_uni:,.2f}")

            if plot:
                split_idx  = int(len(monthly_df_cleaned) * train_size)
                test_dates = monthly_df_cleaned.iloc[split_idx:].index
                full_dates = monthly_df_cleaned.index
                history    = (
                    np.exp(monthly_df_cleaned["price"].values)
                    if type_of_outlier == "log_transform"
                    else monthly_df_cleaned["price"].values
                )
                plt.figure(figsize=(12, 6))
                plt.plot(full_dates, history, color="lightgray", label="History")
                plt.plot(test_dates, actual_uni, color="black", label="Actual")
                plt.plot(
                    test_dates, preds_uni_d, "--",
                    label=f"Univariate LSTM (RMSE=${int(rmse_uni):,})",
                )
                plt.title(f"Univariate LSTM  |  {type_of_outlier}")
                plt.xlabel("Date"); plt.ylabel("Price ($)")
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.show()

        # ── Seq2Seq model ─────────────────────────────────────────────────────
        if "sequential" in models:
            try:
                _, preds_seq, actual_seq, _, test_idx_seq = (
                    run_lstm_univariate_seq2seq(
                        monthly_df_cleaned,
                        train_size=train_size,
                        seq_len=seq_len,
                        horizon=horizon,
                    )
                )
                if type_of_outlier == "log_transform":
                    actual_seq_d = np.exp(actual_seq)
                    preds_seq_d  = np.exp(preds_seq)
                else:
                    actual_seq_d = actual_seq
                    preds_seq_d  = preds_seq

                rmse_seq = calculate_rmse(actual_seq_d, preds_seq_d)
                print(f"  Seq2Seq ({horizon}-step) RMSE = ${rmse_seq:,.2f}")

                if plot:
                    plt.figure(figsize=(12, 6))
                    plt.plot(test_idx_seq, actual_seq_d, color="black", label="Actual")
                    plt.plot(
                        test_idx_seq, preds_seq_d, "--",
                        label=f"Seq2Seq LSTM (RMSE=${int(rmse_seq):,})",
                    )
                    plt.title(f"Seq2Seq LSTM  |  {type_of_outlier}")
                    plt.xlabel("Date"); plt.ylabel("Price ($)")
                    plt.legend(); plt.grid(True, alpha=0.3)
                    plt.tight_layout(); plt.show()

            except ValueError as exc:
                print(f"  Seq2Seq skipped: {exc}")


def lstm_evaluation_pipeline2(
    df: pd.DataFrame,
    pipeline_function,
    add_seasonality_function,
    feature_sets: dict,
    types_of_outlier: tuple = ("none", "remove", "log_transform"),
    train_size: float = 0.85,
    seq_len: int = 12,
    gridsearch: bool = False,
    param_grid: dict = None,
    plot: bool = True,
    explain: bool = True,
) -> None:
    """Evaluate a multivariate autoregressive LSTM across outlier strategies and feature sets.

    For each combination of (outlier strategy × feature set):

    1. Builds monthly features via *pipeline_function* and
       *add_seasonality_function*.
    2. Optionally runs :func:`gridsearch_lstm` to find the best
       hyper-parameters.
    3. Trains :func:`run_lstm_multivariate2`.
    4. Reports RMSE (back-transformed to dollars if log prices were used).
    5. Optionally renders forecast plots and SHAP explanations using
       :class:`shap.GradientExplainer`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame.
    pipeline_function : callable
        Cleaning/aggregation pipeline (see :func:`pipeline`).
    add_seasonality_function : callable
        Seasonality feature builder (see :func:`add_seasonality_features`).
    feature_sets : dict
        Mapping ``{name: [col1, col2, …]}`` of *exogenous* feature columns.
        The price column is always added internally as the autoregressive input.
    types_of_outlier : tuple of str, optional
        Outlier strategies to evaluate.
    train_size : float, optional
        Training fraction.  Default: ``0.85``.
    seq_len : int, optional
        Look-back window in months.  Default: ``12``.
    gridsearch : bool, optional
        If ``True``, call :func:`gridsearch_lstm` before fitting.
        Default: ``False``.
    param_grid : dict, optional
        Hyper-parameter grid forwarded to :func:`gridsearch_lstm`.  Falls back
        to :data:`DEFAULT_LSTM_PARAM_GRID` when ``None``.
    plot : bool, optional
        Render forecast plots.  Default: ``True``.
    explain : bool, optional
        Compute SHAP gradient explanations.  Default: ``True``.
    """
    if param_grid is None:
        param_grid = DEFAULT_LSTM_PARAM_GRID

    for type_of_outlier in types_of_outlier:
        _, _, monthly_df_cleaned = pipeline_function(
            df, type_of_outlier=type_of_outlier, plotting=False
        )
        monthly_df_cleaned = add_seasonality_function(monthly_df_cleaned)
        print(f"\n=== Multivariate LSTM | outlier: {type_of_outlier} ===")

        for feat_name, feature_cols_exog in feature_sets.items():
            print(f"\n  --- features: {feat_name} ---")

            # ── Optional grid search ──────────────────────────────────────────
            best_params: dict = {}
            if gridsearch:
                best, _ = gridsearch_lstm(
                    monthly_df_cleaned,
                    feature_cols=feature_cols_exog,
                    param_grid=param_grid,
                    seq_len=seq_len,
                    train_size=train_size,
                )
                best_params = best[0]
                print(f"  Best params: {best_params}")

            # ── Train model ───────────────────────────────────────────────────
            model, preds, actual, rmse = run_lstm_multivariate2(
                monthly_df_cleaned,
                feature_cols_exog=feature_cols_exog,
                train_size=train_size,
                seq_len=seq_len,
                **best_params,
            )

            # ── Back-transform if log prices ──────────────────────────────────
            if type_of_outlier == "log_transform":
                actual_dollar  = np.exp(actual)
                preds_dollar   = np.exp(preds)
                rmse_dollar    = calculate_rmse(actual_dollar, preds_dollar)
                history_dollar = np.exp(monthly_df_cleaned["price"].values)
                print(f"  RMSE = ${rmse_dollar:,.2f}")
            else:
                actual_dollar  = actual
                preds_dollar   = preds
                rmse_dollar    = rmse
                history_dollar = monthly_df_cleaned["price"].values
                print(f"  RMSE = ${rmse:,.2f}")

            split_idx  = int(len(monthly_df_cleaned) * train_size)
            test_dates = monthly_df_cleaned.iloc[split_idx:].index
            full_dates = monthly_df_cleaned.index

            # ── Forecast plot ─────────────────────────────────────────────────
            if plot:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    full_dates, history_dollar,
                    color="lightgray", label="Actual History", linewidth=2,
                )
                plt.plot(
                    test_dates, actual_dollar,
                    color="black", label="Actual Test", linewidth=2,
                )
                plt.plot(
                    test_dates, preds_dollar, "--",
                    label=f"LSTM ({feat_name}) (RMSE=${int(rmse_dollar):,})",
                    linewidth=2,
                )
                plt.title(
                    f"LSTM Predictions  |  {type_of_outlier}  |  {feat_name}"
                )
                plt.xlabel("Date"); plt.ylabel("Price ($)")
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.show()

            # ── SHAP explanation ──────────────────────────────────────────────
            if explain:
                print("  Computing SHAP values …")
                target            = "price"
                feature_cols_full = [target] + feature_cols_exog
                train_df_s        = monthly_df_cleaned.iloc[:split_idx]
                test_df_s         = monthly_df_cleaned.iloc[split_idx:]

                sx = MinMaxScaler()
                sy = MinMaxScaler()
                sx.fit(train_df_s[feature_cols_full])
                sy.fit(train_df_s[[target]])

                train_X = sx.transform(train_df_s[feature_cols_full])
                test_X  = sx.transform(test_df_s[feature_cols_full])
                train_y = sy.transform(train_df_s[[target]])

                X_train_seq, _ = create_sequences(train_X, train_y, seq_len)

                X_all_seq, _ = create_sequences(
                    np.vstack([train_X, test_X]),
                    sy.transform(monthly_df_cleaned[[target]]),
                    seq_len,
                )
                X_test_seq = X_all_seq[split_idx - seq_len:]

                bg_idx     = np.random.choice(
                    len(X_train_seq),
                    size=min(100, len(X_train_seq)),
                    replace=False,
                )
                background = X_train_seq[bg_idx]

                explainer  = shap.GradientExplainer(model, background)
                shap_raw   = explainer.shap_values(X_test_seq)

                if isinstance(shap_raw, list):
                    shap_raw = shap_raw[0]
                shap_raw = np.array(shap_raw)
                if shap_raw.ndim > 3:
                    shap_raw = np.squeeze(shap_raw)

                shap_abs   = (
                    np.abs(shap_raw).mean(axis=1)
                    if shap_raw.ndim == 3
                    else np.abs(shap_raw)
                )
                mean_shap     = shap_abs.mean(axis=0)
                feature_names = np.array(feature_cols_full)
                sorted_idx    = np.argsort(mean_shap)[::-1]

                # Bar chart of mean |SHAP| per feature
                plt.figure(figsize=(8, 4))
                sns.barplot(
                    x=mean_shap[sorted_idx],
                    y=feature_names[sorted_idx],
                    palette="magma",
                )
                plt.title(
                    f"LSTM – Mean |SHAP|  |  {type_of_outlier}  |  {feat_name}"
                )
                plt.xlabel("Mean |SHAP value|")
                plt.tight_layout()
                plt.show()

                # SHAP beeswarm on the last timestep of each test sequence
                if shap_raw.ndim == 3:
                    shap_last = shap_raw[:, -1, :]
                    X_last    = X_test_seq[:, -1, :]
                    shap.summary_plot(
                        shap_last, X_last,
                        feature_names=feature_names.tolist(),
                        show=False,
                    )
                    plt.title(
                        f"SHAP summary (last timestep)  |  "
                        f"{type_of_outlier}  |  {feat_name}"
                    )
                    plt.tight_layout()
                    plt.show()
