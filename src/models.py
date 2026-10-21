"""Classical ML modelling (Task 2C) for Northgate AI Stock Predictor.

Predicts next-trading-day log returns from the leakage-free Task 2B
feature files. Classical models only: zero-return baseline, Linear
Regression, Ridge, Random Forest, XGBoost, SVR. No deep learning.

Protocol (per equity, deterministic, seed 42):
  1. Load supervised Parquet, sort by date, drop unusable rows.
  2. Chronological 80/20 split with a 1-row embargo: first 80% minus its
     last row = development (so no dev target uses a test-period price),
     last 20% = untouched final test (never shuffled; test strictly later).
  3. Hyperparameter tuning with sklearn TimeSeriesSplit (5 folds, gap=1)
     INSIDE development data only, so no training target touches its
     validation block. Scalers live inside Pipelines, hence are fit
     on each training fold only (never on validation/test).
  4. Refit the CV-selected config on all development data.
  5. Evaluate ONCE on the final test set.

Metrics: RMSE, MAE, MAPE (safe: exact-zero actuals excluded, documented),
R2, directional accuracy with the deterministic convention
sign(pred)==sign(actual) where sign(0)==0 (exact 0-vs-0 counts correct;
no thresholds introduced).

MAPE LIMITATION: the target is a log return that can sit near zero, so
MAPE is unstable/misleading here; RMSE and MAE are the primary metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.features import EQUITY_FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_FRACTION = 0.2  # final chronological 20% is the untouched test set
N_SPLITS = 5  # TimeSeriesSplit folds inside development data

FEATURES_DIR = Path("data/processed/features")
MODELS_DIR = Path("data/processed/models")

MODEL_NAMES = [
    "random_walk",
    "linear_regression",
    "ridge",
    "random_forest",
    "xgboost",
    "svr",
]

# Production grids: genuinely modest for a college research project.
# Fits per ticker = (6 ridge + 4 RF + 2 XGB + 3 SVR) x 5 folds = 75.
# Leakage protection is unchanged: TimeSeriesSplit on dev data only,
# scalers inside Pipelines, test set never touched during tuning.
RIDGE_GRID = {"ridge__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
RF_GRID = {
    "n_estimators": [200],
    "max_depth": [5, 10],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt"],
}
XGB_GRID = {
    "n_estimators": [200],
    "max_depth": [3, 5],
    "learning_rate": [0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
SVR_GRID = {
    "svr__C": [0.1, 1.0, 10.0],
    "svr__epsilon": [0.01],
    "svr__gamma": ["scale"],
    "svr__kernel": ["rbf"],
}

# Lightweight grids for unit tests only: same code path and model set,
# tiny searches so the suite finishes quickly. Never used in production.
TEST_GRIDS = {
    "random_walk": None,
    "linear_regression": None,
    "ridge": {"ridge__alpha": [0.1, 1.0]},
    "random_forest": {"n_estimators": [10], "max_depth": [3, None]},
    "xgboost": {"n_estimators": [10], "max_depth": [2],
                "learning_rate": [0.05], "subsample": [0.8],
                "colsample_bytree": [0.8]},
    "svr": {"svr__C": [0.1, 1.0], "svr__epsilon": [0.01],
            "svr__gamma": ["scale"], "svr__kernel": ["rbf"]},
}


# Pure helpers (unit-tested)

def chronological_split(df: pd.DataFrame,
                        test_fraction: float = TEST_FRACTION,
                        date_col: str = "date",
                        embargo: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split with a horizon embargo for the 1-day-ahead target.

    Because row t carries target log(adj_{t+1}/adj_t), the last development
    row's target would otherwise consume the first test row's price. Hence
    ``embargo`` rows (default 1 = the one-step horizon) between dev and test
    are left entirely unused: dev = rows [:cut-embargo], test = rows [cut:].
    Sorts ascending, never shuffles. Test dates are strictly later, and no
    development target endpoint reaches into the test period.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    cut = int(n * (1.0 - test_fraction))
    dev = df.iloc[:cut - embargo].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    if len(dev) == 0 or len(test) == 0:
        raise ValueError(
            f"not enough rows for split with embargo (n={n}, "
            f"test_fraction={test_fraction}, embargo={embargo})")
    assert dev[date_col].max() < test[date_col].min(), "split must be ordered"
    return dev, test


def split_X_y(df: pd.DataFrame, feature_cols: list[str] = EQUITY_FEATURE_COLUMNS,
              target_col: str = TARGET_COLUMN):
    """Separate feature matrix X from target vector y."""
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def zero_baseline_predict(n: int) -> np.ndarray:
    """Random-walk baseline: predicted log return = 0 for every row."""
    return np.zeros(n, dtype=float)


def get_time_series_cv(n_splits: int = N_SPLITS,
                       gap: int = 1) -> TimeSeriesSplit:
    """Expanding-window chronological CV splitter (no shuffling).

    ``gap=1`` leaves one unused row between each training block and its
    validation block, so a training row's one-day-ahead target (which uses
    the next row's price) can never touch validation-period information.
    """
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def compute_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, safe MAPE, R2, directional accuracy (deterministic).

    MAPE: rows with actual == 0 exactly are excluded (no division by
    zero); NaN if no nonzero actuals. Zero-direction convention:
    np.sign (sign(0)==0), so exact 0-vs-0 counts as correct.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nonzero = y_true != 0.0
    mape = (float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero])
                                 / y_true[nonzero])) * 100.0)
            if nonzero.any() else float("nan"))
    r2 = float(r2_score(y_true, y_pred))
    direction = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2,
            "directional_accuracy": direction}


def build_estimators() -> dict:
    """Model estimators; Ridge/SVR wrap StandardScaler INSIDE a Pipeline."""
    return {
        "random_walk": None,  # no training; predicts 0
        "linear_regression": LinearRegression(),
        "ridge": Pipeline([("scaler", StandardScaler()),
                           ("ridge", Ridge(random_state=RANDOM_STATE))]),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE,
                                               n_jobs=1),  # single-threaded:
        # joblib process pools can deadlock in restricted sandboxes; n_jobs=1
        # is also the strictest determinism choice.
        "xgboost": XGBRegressor(random_state=RANDOM_STATE, n_jobs=1,
                                verbosity=0),
        "svr": Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
    }


def build_grids(test_mode: bool = False) -> dict:
    """Param grids keyed by model name (None = no tuning).

    Production (default): the modest grids above. Test mode: the tiny
    TEST_GRIDS, exercising the identical tuning code path cheaply.
    """
    if test_mode:
        return dict(TEST_GRIDS)
    return {
        "random_walk": None,
        "linear_regression": None,
        "ridge": RIDGE_GRID,
        "random_forest": RF_GRID,
        "xgboost": XGB_GRID,
        "svr": SVR_GRID,
    }


def tune_on_development(X_dev: np.ndarray, y_dev: np.ndarray,
                        estimator, param_grid: dict | None,
                        model_name: str) -> tuple:
    """Grid-search with TimeSeriesSplit on development data ONLY.

    Returns (best_estimator_unfitted_or_fitted, best_params, cv_rows)
    where cv_rows records per-combination mean CV MSE for the audit trail.
    The caller refits the winner on all development data.
    """
    cv = get_time_series_cv()
    if param_grid is None:
        return estimator, {}, []
    search = GridSearchCV(estimator, param_grid, cv=cv,
                          scoring="neg_mean_squared_error", n_jobs=1)
    search.fit(X_dev, y_dev)  # dev data only; test set never seen here
    rows = []
    for params, score in zip(search.cv_results_["params"],
                             search.cv_results_["mean_test_score"]):
        rows.append({"model": model_name,
                     "params": str(params),
                     "mean_cv_neg_mse": float(score),
                     "mean_cv_rmse": float(np.sqrt(-score))})
    return search.best_estimator_, dict(search.best_params_), rows


# Per-ticker pipeline

def run_ticker(ticker: str, df: pd.DataFrame,
               grids: dict | None = None) -> tuple[list, list, list, list]:
    """Dev/test split, tune on dev, refit, evaluate once on test.

    Returns (metric_rows, cv_rows, prediction_rows, selected_rows).
    ``grids`` defaults to the production grids; tests inject TEST_GRIDS.
    """
    df = df.sort_values("date").reset_index(drop=True)
    dev, test = chronological_split(df)
    X_dev, y_dev = split_X_y(dev)
    X_test, y_test = split_X_y(test)
    test_dates = test["date"]

    estimators = build_estimators()
    grids = build_grids() if grids is None else grids
    metric_rows, cv_rows, pred_rows, selected_rows = [], [], [], []

    for name in MODEL_NAMES:
        if name == "random_walk":
            preds = zero_baseline_predict(len(test))
            best_params, cv_rmse = {}, float("nan")
        else:
            best_est, best_params, cv_combo_rows = tune_on_development(
                X_dev, y_dev, estimators[name], grids[name], name)
            for r in cv_combo_rows:
                cv_rows.append({"ticker": ticker, **r})
            cv_rmse = min(r["mean_cv_rmse"] for r in cv_combo_rows) \
                if cv_combo_rows else float("nan")
            best_est.fit(X_dev, y_dev)  # refit on full development period
            preds = np.asarray(best_est.predict(X_test), dtype=float)
        metrics = compute_metrics(y_test, preds)
        metric_rows.append({"ticker": ticker, "model": name, **metrics})
        selected_rows.append({"ticker": ticker, "model": name,
                              "best_params": str(best_params),
                              "dev_cv_rmse": cv_rmse,
                              "n_dev": len(dev), "n_test": len(test)})
        for d, a, p in zip(test_dates, y_test, preds):
            pred_rows.append({"date": pd.to_datetime(d), "ticker": ticker,
                              "model": name, "actual_return": float(a),
                              "predicted_return": float(p)})
        logger.info("%s/%s test RMSE=%.6f DA=%.3f", ticker, name,
                    metrics["rmse"], metrics["directional_accuracy"])
    return metric_rows, cv_rows, pred_rows, selected_rows



def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load equity universe from config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_filename(ticker: str) -> str:
    """Strip '^' for safe filenames."""
    return ticker.replace("^", "")


def run_models(config_path: str | Path = "config.yaml",
               features_dir: str | Path = FEATURES_DIR,
               output_dir: str | Path = MODELS_DIR,
               grids: dict | None = None) -> dict:
    """Train/evaluate all models for all equities (deterministic, offline)."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    tickers: list[str] = list(load_config(config_path)["universe"]["tickers"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = build_grids() if grids is None else grids

    all_metrics, all_cv, all_preds, all_sel = [], [], [], []
    for ticker in tickers:
        path = Path(features_dir) / (sanitize_filename(ticker) + ".parquet")
        df = pd.read_parquet(path)
        m, c, p, s = run_ticker(ticker, df, grids=grids)
        all_metrics += m
        all_cv += c
        all_preds += p
        all_sel += s

    pd.DataFrame(all_metrics).to_csv(out_dir / "model_metrics.csv",
                                     index=False)
    pd.DataFrame(all_cv).to_csv(out_dir / "model_cv_results.csv", index=False)
    pd.DataFrame(all_preds).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(all_sel).to_csv(out_dir / "selected_hyperparameters.csv",
                                 index=False)

    audit = leakage_audit(all_preds, all_sel)
    print("=" * 72)
    print("TASK 2C COMPLETE — classical ML (test set evaluated once)")
    print(f"Outputs -> {out_dir}")
    print("LEAKAGE AUDIT:")
    for line in audit:
        print(f"  {line}")
    print("=" * 72)
    return {"output_dir": str(out_dir), "audit": audit,
            "n_predictions": len(all_preds)}


def leakage_audit(pred_rows: list, selected_rows: list) -> list[str]:
    """Deterministic audit lines proving test isolation and chronology."""
    preds = pd.DataFrame(pred_rows)
    checks = []
    ok = True
    for ticker, g in preds.groupby("ticker"):
        base = g[g["model"] == "random_walk"][["date"]].rename(
            columns={"date": "d0"})
        others = g[g["model"] != "random_walk"]
        same_dates = others.groupby("model")["date"].apply(
            lambda s: set(s) == set(base["d0"]))
        ok = ok and bool(same_dates.all())
    checks.append(f"all models share identical test dates per ticker: {ok}")
    n_test_models = sorted(preds["model"].unique().tolist())
    checks.append(f"models evaluated: {n_test_models}")
    checks.append("scaler inside Pipeline for ridge/svr: "
                  f"{'scaler' in build_estimators()['ridge'].named_steps} / "
                  f"{'scaler' in build_estimators()['svr'].named_steps}")
    checks.append(f"split rule: chronological 80/20 + 1-row embargo "
                  f"(test_fraction={TEST_FRACTION}); "
                  f"TimeSeriesSplit(n_splits={N_SPLITS}, gap=1) on dev only; "
                  f"seed={RANDOM_STATE}")
    checks.append("hyperparameters selected on dev CV only; refit on dev; "
                  "test predicted once")
    return checks


def main(config_path: str | Path = "config.yaml") -> dict:
    """CLI entry point: ``python -m src.models``."""
    return run_models(config_path=config_path)


if __name__ == "__main__":
    main()
