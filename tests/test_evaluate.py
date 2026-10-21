"""Tests for src.evaluate (synthetic fixtures; no training, no internet)."""

import numpy as np
import pandas as pd
import pytest

from src import evaluate
from src.evaluate import (
    aggregate_summary,
    baseline_comparison,
    best_model_by_ticker,
    run_evaluation,
    validate_outputs,
)
from src.models import MODEL_NAMES, compute_metrics


def make_metrics():
    """Small stored-metrics frame: 2 tickers x all 6 models."""
    rng = np.random.default_rng(0)
    rows = []
    for t in ("AAA", "BBB"):
        for i, m in enumerate(MODEL_NAMES):
            rows.append({"ticker": t, "model": m,
                         "rmse": 0.010 + 0.001 * i + rng.normal(0, 1e-4),
                         "mae": 0.008, "mape": 150.0,
                         "r2": -0.02 * i,
                         "directional_accuracy": 0.50 + 0.005 * i})
    # force known winner: linear beats baseline on AAA
    rows[1]["rmse"] = 0.005
    return pd.DataFrame(rows)


def make_preds(tickers=("AAA", "BBB"), n=40, seed=1):
    """Synthetic prediction-level rows with identical dates per model."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for t in tickers:
        actual = rng.normal(0, 0.01, n)
        for m in MODEL_NAMES:
            pred = actual + rng.normal(0, 0.005, n) if m != "random_walk" \
                else np.zeros(n)
            for d, a, p in zip(dates, actual, pred):
                rows.append({"date": d, "ticker": t, "model": m,
                             "actual_return": a, "predicted_return": p})
    return pd.DataFrame(rows)


def write_models_dir(path, metrics, preds=None):
    """Write a fake Task 2C outputs directory."""
    path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path / "model_metrics.csv", index=False)
    if preds is not None:
        preds.to_csv(path / "predictions.csv", index=False)


def write_config(path, tickers=("AAA", "BBB")):
    import yaml
    cfg = {"universe": {"tickers": list(tickers)}}
    p = path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# 1-5. metric calculations reuse Task 2C definitions exactly
def test_1_5_metrics_match_models_definitions():
    assert evaluate.compute_metrics is compute_metrics  # same function
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    m = compute_metrics(y_true, y_pred)
    assert m["rmse"] == np.sqrt(0.25)          # 2. RMSE
    assert m["mae"] == 0.25                    # 3. MAE
    assert m["r2"] == 1 - 1.0 / 5.0            # 4. R2
    d = compute_metrics(np.array([.01, -.02, 0., -.01]),
                        np.array([.02, .03, 0., -.005]))
    assert d["directional_accuracy"] == 0.75    # 5. directional accuracy


# 6. MAPE zero handling preserved
def test_6_mape_zero_handling():
    m = compute_metrics(np.array([0.0, 0.02, -0.04]),
                        np.array([0.01, 0.02, -0.02]))
    assert np.isfinite(m["mape"])
    assert np.isnan(compute_metrics(np.zeros(3), np.zeros(3))["mape"])


# 7. baseline RMSE improvement math
def test_7_baseline_improvement():
    comp = baseline_comparison(make_metrics())
    aaa_lin = comp[(comp["ticker"] == "AAA") &
                   (comp["model"] == "linear_regression")].iloc[0]
    base = comp[(comp["ticker"] == "AAA") &
                (comp["model"] == "random_walk")].iloc[0]
    assert aaa_lin["rmse_improvement"] > 0  # better than baseline
    assert base["rmse_improvement"] == 0.0  # baseline vs itself
    assert aaa_lin["rmse_abs_diff"] == base["rmse"] - aaa_lin["rmse"]


# 8. best-model selection = lowest RMSE, no tuning involved
def test_8_best_model_selection():
    metrics = make_metrics()
    comp = baseline_comparison(metrics)
    best = best_model_by_ticker(metrics, comp)
    assert set(best["ticker"]) == {"AAA", "BBB"}
    aaa = best[best["ticker"] == "AAA"].iloc[0]
    assert aaa["best_model"] == "linear_regression"
    assert aaa["rmse"] == metrics[(metrics["ticker"] == "AAA")][
        "rmse"].min()


# 9. aggregate summary math (per-ticker metrics, never pooled predictions)
def test_9_aggregate_summary():
    metrics = make_metrics()
    comp = baseline_comparison(metrics)
    agg = aggregate_summary(metrics, comp)
    assert set(agg["model"]) == set(MODEL_NAMES)
    rw = agg[agg["model"] == "random_walk"].iloc[0]
    assert rw["mean_rmse"] == metrics[metrics["model"] == "random_walk"][
        "rmse"].mean()
    assert rw["n_tickers"] == 2
    lin = agg[agg["model"] == "linear_regression"].iloc[0]
    assert lin["n_tickers_beat_baseline"] >= 1


# 10. missing model/ticker combinations fail loudly
def test_10_missing_combination_loud():
    metrics = make_metrics()
    metrics = metrics[~((metrics["ticker"] == "AAA") &
                        (metrics["model"] == "svr"))]
    with pytest.raises(ValueError, match="missing ticker/model"):
        validate_outputs(metrics, None, ["AAA", "BBB"])
    norw = make_metrics()
    norw = norw[norw["model"] != "random_walk"]
    with pytest.raises(ValueError, match="random_walk"):
        validate_outputs(norw, None, ["AAA", "BBB"])


# 11. inconsistent test dates fail loudly; duplicates fail loudly
def test_11_inconsistent_dates_loud():
    preds = make_preds()
    bad = preds.copy()
    mask = ((bad["ticker"] == "AAA") & (bad["model"] == "ridge"))
    bad = bad[~(mask & (bad["date"] == bad["date"].min()))]
    with pytest.raises(ValueError, match="different test observations"):
        validate_outputs(make_metrics(), bad, ["AAA", "BBB"])
    dup = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_outputs(make_metrics(), dup, ["AAA", "BBB"])


# 12. evaluator does not modify raw data
def test_12_raw_untouched(tmp_path):
    import hashlib
    from pathlib import Path
    raw = Path("data/raw")
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(raw.rglob("*.parquet"))}
    models_dir = tmp_path / "models"
    write_models_dir(models_dir, make_metrics(), make_preds())
    cfg = write_config(tmp_path)
    run_evaluation(config_path=cfg, models_dir=models_dir,
                   output_dir=tmp_path / "eval")
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(raw.rglob("*.parquet"))}
    assert before and before == after


# 13. deterministic output for identical inputs
def test_13_deterministic(tmp_path):
    models_dir = tmp_path / "models"
    write_models_dir(models_dir, make_metrics(), make_preds())
    cfg = write_config(tmp_path)
    run_evaluation(config_path=cfg, models_dir=models_dir,
                   output_dir=tmp_path / "e1")
    run_evaluation(config_path=cfg, models_dir=models_dir,
                   output_dir=tmp_path / "e2")
    for f in ("evaluation_metrics.csv", "baseline_comparison.csv",
              "best_model_by_ticker.csv", "aggregate_model_summary.csv",
              "evaluation_summary.txt"):
        assert (tmp_path / "e1" / f).read_bytes() == \
            (tmp_path / "e2" / f).read_bytes()


def test_missing_models_dir_loud(tmp_path):
    cfg = write_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="model_metrics.csv"):
        run_evaluation(config_path=cfg, models_dir=tmp_path / "nope",
                       output_dir=tmp_path / "eval")


def test_stored_only_path_documents_limitation(tmp_path):
    models_dir = tmp_path / "models"
    write_models_dir(models_dir, make_metrics(), preds=None)
    cfg = write_config(tmp_path)
    out = run_evaluation(config_path=cfg, models_dir=models_dir,
                         output_dir=tmp_path / "eval")
    m = pd.read_csv(tmp_path / "eval" / "evaluation_metrics.csv")
    assert (m["metric_source"] == "stored").all()
    assert out["n_tickers"] == 2
    assert (tmp_path / "eval" / "plots").glob("*.png")
