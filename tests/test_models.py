"""Tests for src.models (synthetic/small data; no internet)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src import models
from src.features import EQUITY_FEATURE_COLUMNS, TARGET_COLUMN
from src.models import (
    MODEL_NAMES,
    TEST_GRIDS,
    build_estimators,
    build_grids,
    chronological_split,
    compute_metrics,
    get_time_series_cv,
    run_models,
    run_ticker,
    split_X_y,
    tune_on_development,
    zero_baseline_predict,
)


def make_supervised(n=150, start="2020-01-01", seed=3):
    """Tiny supervised-style frame with all feature columns + target."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    data = {"date": dates, "ticker": "SYN"}
    for c in EQUITY_FEATURE_COLUMNS:
        data[c] = rng.normal(0, 1, n)
    data[TARGET_COLUMN] = rng.normal(0, 0.01, n)
    return pd.DataFrame(data)


# 1. chronological split sizes (80/20 minus the 1-row horizon embargo)
def test_1_chronological_split_sizes():
    df = make_supervised(100)
    dev, test = chronological_split(df, test_fraction=0.2)
    assert len(dev) == 79 and len(test) == 20


# 2. no random shuffling: order preserved (row 79 is the embargo gap)
def test_2_no_shuffling():
    df = make_supervised(100)
    dev, test = chronological_split(df)
    assert dev["date"].is_monotonic_increasing
    assert test["date"].is_monotonic_increasing
    assert list(dev["date"]) == list(df["date"].iloc[:79])
    assert list(test["date"]) == list(df["date"].iloc[80:])
    assert df["date"].iloc[79] not in set(dev["date"])
    assert df["date"].iloc[79] not in set(test["date"])


# 3. train dates strictly before test dates
def test_3_train_before_test():
    df = make_supervised(100)
    dev, test = chronological_split(df)
    assert dev["date"].max() < test["date"].min()


# 4. target/feature separation
def test_4_split_X_y():
    df = make_supervised(50)
    X, y = split_X_y(df)
    assert X.shape == (50, len(EQUITY_FEATURE_COLUMNS))
    assert y.shape == (50,)
    assert TARGET_COLUMN not in EQUITY_FEATURE_COLUMNS


# 5. random-walk baseline returns zero
def test_5_baseline_zeros():
    preds = zero_baseline_predict(7)
    assert (preds == 0.0).all() and len(preds) == 7


# 6. metric calculations on known values
def test_6_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    m = compute_metrics(y_true, y_pred)
    assert m["rmse"] == np.sqrt(0.25)
    assert m["mae"] == 0.25
    # r2 hand-check via sklearn convention
    ss_res, ss_tot = 1.0, 5.0
    assert m["r2"] == 1 - ss_res / ss_tot


# 7. directional accuracy incl. deterministic zero convention
def test_7_directional_accuracy():
    y_true = np.array([0.01, -0.02, 0.0, -0.01])
    y_pred = np.array([0.02, 0.03, 0.0, -0.005])
    m = compute_metrics(y_true, y_pred)
    # +, -, 0-vs-0, -  -> 3/4 correct
    assert m["directional_accuracy"] == 0.75


# 8. MAPE zero/near-zero handling
def test_8_mape_zero_handling():
    y_true = np.array([0.0, 0.02, -0.04])
    y_pred = np.array([0.01, 0.02, -0.02])
    m = compute_metrics(y_true, y_pred)
    expected = (0.0 + 0.5) / 2 * 100.0  # zero-actual row excluded
    assert np.isfinite(m["mape"]) and m["mape"] == expected
    m_all0 = compute_metrics(np.zeros(3), np.zeros(3))
    assert np.isnan(m_all0["mape"])  # documented NaN, never inf/crash


# 9. StandardScaler inside Ridge/SVR pipelines
def test_9_scaler_in_pipeline():
    est = build_estimators()
    assert "scaler" in est["ridge"].named_steps
    assert "scaler" in est["svr"].named_steps
    assert type(est["ridge"].named_steps["scaler"]).__name__ == (
        "StandardScaler")
    assert type(est["svr"].named_steps["scaler"]).__name__ == (
        "StandardScaler")


# 10. deterministic seeds + repeat-fit stability
def test_10_deterministic_seeds():
    est = build_estimators()
    assert est["random_forest"].get_params()["random_state"] == 42
    assert est["xgboost"].get_params()["random_state"] == 42
    assert est["ridge"].get_params()["ridge__random_state"] == 42
    df = make_supervised(120)
    dev, _ = chronological_split(df)
    X, y = split_X_y(dev)
    rf1 = est["random_forest"].__class__(random_state=42)
    rf1.fit(X, y)
    rf2 = est["random_forest"].__class__(random_state=42)
    rf2.fit(X, y)
    assert (rf1.predict(X) == rf2.predict(X)).all()


# 11. expected model names
def test_11_model_names():
    assert MODEL_NAMES == ["random_walk", "linear_regression", "ridge",
                           "random_forest", "xgboost", "svr"]


# 12+15. output schemas + prediction identity via lightweight pipeline run
# (tiny TEST_GRIDS: same code path, fraction of the production cost)
def test_12_15_output_schema_and_identity(tmp_path):
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for t in ("AAA", "BBB"):
        df = make_supervised(150, seed=abs(hash(t)) % 1000)
        df["ticker"] = t
        df.to_parquet(feat_dir / f"{t}.parquet", index=False)
    import yaml
    cfg = {"universe": {"tickers": ["AAA", "BBB"]}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = run_models(config_path=cfg_path, features_dir=feat_dir,
                     output_dir=tmp_path / "models", grids=TEST_GRIDS)
    metrics = pd.read_csv(tmp_path / "models" / "model_metrics.csv")
    preds = pd.read_csv(tmp_path / "models" / "predictions.csv")
    cv = pd.read_csv(tmp_path / "models" / "model_cv_results.csv")
    sel = pd.read_csv(tmp_path / "models" / "selected_hyperparameters.csv")
    assert list(metrics.columns) == ["ticker", "model", "rmse", "mae",
                                     "mape", "r2", "directional_accuracy"]
    assert list(preds.columns) == ["date", "ticker", "model",
                                   "actual_return", "predicted_return"]
    assert set(preds["model"]) == set(MODEL_NAMES)
    assert set(preds["ticker"]) == {"AAA", "BBB"}
    assert len(preds) == 2 * len(MODEL_NAMES) * 30  # 20% of 150
    assert {"ticker", "model", "mean_cv_rmse"}.issubset(cv.columns)
    assert {"ticker", "model", "best_params"}.issubset(sel.columns)
    assert out["n_predictions"] == len(preds)


# 13. no test data in hyperparameter selection: corrupting test targets
# must not change selected hyperparameters (lightweight TEST_GRIDS search)
def test_13_test_isolation():
    df = make_supervised(150)
    _, _, _, sel_clean = run_ticker("SYN", df, grids=TEST_GRIDS)
    corrupted = df.copy()
    dev_n = int(len(df) * 0.8)
    corrupted.loc[corrupted.index[dev_n:], TARGET_COLUMN] *= -999.0
    _, _, _, sel_corr = run_ticker("SYN", corrupted, grids=TEST_GRIDS)
    params_clean = {r["model"]: r["best_params"] for r in sel_clean}
    params_corr = {r["model"]: r["best_params"] for r in sel_corr}
    assert params_clean == params_corr


# production grids stay genuinely modest (college-scale search budget)
def test_production_grids_modest():
    import itertools
    grids = build_grids()
    assert set(grids) == set(MODEL_NAMES)
    combos = {m: (1 if g is None else
                  len(list(itertools.product(*g.values()))))
              for m, g in grids.items()}
    assert combos["ridge"] <= 6
    assert combos["random_forest"] <= 6
    assert combos["xgboost"] <= 4
    assert combos["svr"] <= 6
    # test grids are a strict subset-scale config, same model keys
    assert set(TEST_GRIDS) == set(MODEL_NAMES)
    assert build_grids(test_mode=True) == TEST_GRIDS


# 14. TimeSeriesSplit is chronological (train before validation, expanding)
def test_14_tss_chronology():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    cv = get_time_series_cv(n_splits=5)
    splits = list(cv.split(X, y))
    assert len(splits) == 5
    prev_valid_start = -1
    for train_idx, valid_idx in splits:
        assert train_idx.max() < valid_idx.min()  # validation after train
        assert valid_idx.min() > prev_valid_start  # folds move forward
        prev_valid_start = valid_idx.min()


# REGRESSION 1: final development target must not reach into test period.
# Target is linked to a known price path so the endpoint is provable.
def test_r1_dev_target_no_test_overlap():
    n = 100
    adj = 100.0 + np.arange(n, dtype=float)
    df = make_supervised(n)
    df[TARGET_COLUMN] = np.append(np.log(adj[1:] / adj[:-1]), np.nan)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    assert len(df) == n - 1
    dev, test = chronological_split(df)
    endpoint_pos = len(dev)  # target of last dev row uses next row's price
    endpoint_date = df["date"].iloc[endpoint_pos]
    assert endpoint_date not in set(test["date"])
    assert endpoint_date < test["date"].min()
    # linkage proof: dev's last target used exactly that endpoint price
    assert dev[TARGET_COLUMN].iloc[-1] == np.log(
        adj[endpoint_pos] / adj[endpoint_pos - 1])


# REGRESSION 2: every CV fold leaves the one-observation horizon gap.
def test_r2_cv_fold_horizon_gap():
    X = np.arange(60).reshape(-1, 1)
    y = np.arange(60, dtype=float)
    splits = list(get_time_series_cv(n_splits=5).split(X, y))
    assert len(splits) == 5
    for train_idx, valid_idx in splits:
        assert train_idx.max() < valid_idx.min()
        # exactly one unused row between train end and validation start
        assert valid_idx.min() - train_idx.max() == 2


# REGRESSION 3: corrupting the FIRST test observation (features and target)
# cannot change development rows or CV-selected hyperparameters.
def test_r3_first_test_row_isolation():
    df = make_supervised(150)
    dev_before, test_before = chronological_split(df)
    _, _, _, sel_before = run_ticker("SYN", df, grids=TEST_GRIDS)
    first_test_date = test_before["date"].iloc[0]
    corrupted = df.copy()
    mask = corrupted["date"] == first_test_date
    assert mask.sum() == 1
    for c in EQUITY_FEATURE_COLUMNS + [TARGET_COLUMN]:
        corrupted.loc[mask, c] *= -999.0
    dev_after, _ = chronological_split(corrupted)
    pdt.assert_frame_equal(dev_before, dev_after)
    _, _, _, sel_after = run_ticker("SYN", corrupted, grids=TEST_GRIDS)
    assert ({r["model"]: r["best_params"] for r in sel_before} ==
            {r["model"]: r["best_params"] for r in sel_after})


# tune helper itself only ever receives dev arrays (contract check)
def test_tune_uses_passed_arrays_only():
    df = make_supervised(120)
    dev, test = chronological_split(df)
    X_dev, y_dev = split_X_y(dev)
    assert len(X_dev) == 95 and len(dev) + len(test) == 119  # 120 - embargo
    est, params, rows = tune_on_development(
        X_dev, y_dev, build_estimators()["ridge"],
        {"ridge__alpha": [0.1, 1.0]}, "ridge")
    assert set(params) == {"ridge__alpha"}
    assert len(rows) == 2 and all("mean_cv_rmse" in r for r in rows)
