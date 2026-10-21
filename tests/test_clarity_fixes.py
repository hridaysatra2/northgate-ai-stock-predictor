"""Tests for small clarity fixes (Task 4). No network, no retraining."""

from pathlib import Path

import pandas as pd

from src import eda, evaluate, features, recommendation as rec
from src.portfolio_frontier import efficient_frontier, key_portfolios
import numpy as np


def test_best_model_warning_in_summary(tmp_path):
    metrics = pd.DataFrame([
        {"ticker": "A", "model": "m1", "rmse": 0.02, "mae": 0.01,
         "mape": 1.0, "r2": 0.0, "directional_accuracy": 0.5,
         "metric_source": "recomputed"},
        {"ticker": "A", "model": "random_walk", "rmse": 0.01, "mae": 0.01,
         "mape": 1.0, "r2": 0.0, "directional_accuracy": 0.5,
         "metric_source": "recomputed"},
    ])
    comp = pd.DataFrame([
        {"ticker": "A", "model": "m1", "rmse_improvement": -1.0},
        {"ticker": "A", "model": "random_walk", "rmse_improvement": 0.0},
    ])
    best = evaluate.best_model_by_ticker(metrics, comp)
    agg = evaluate.aggregate_summary(metrics, comp)
    out = tmp_path / "evaluation_summary.txt"
    evaluate.write_summary(metrics, comp, best, agg, out)
    text = out.read_text(encoding="utf-8")
    assert "must not be used for live model selection" in text
    assert "TEST-set" in text


def test_dl_rmse_caption_has_log_return_units():
    src = Path("app.py").read_text(encoding="utf-8")
    assert "RMSE (log-return units)" in src


def test_drawdown_series_removed():
    assert not hasattr(eda, "drawdown_series")
    assert hasattr(eda, "maximum_drawdown")


def test_thresholds_from_dict_no_dead_keyerror():
    assert rec.thresholds_from_dict({}).positive_return_threshold == 0.01
    try:
        rec.thresholds_from_dict("not-a-dict")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_metadata_constant_filename(tmp_path):
    from src.features import METADATA_PATH, run_features
    from pathlib import Path as _P
    raw = _P("data/raw/market")
    run_features(raw_dir=raw, output_dir=tmp_path)
    assert (tmp_path / METADATA_PATH.name).exists()


def test_build_market_features_returns_tuple():
    assert "tuple[pd.DataFrame, pd.DataFrame]" in (
        features.build_market_features.__annotations__.get("return", "")
    )


def test_frontier_solver_options_type_error():
    mu = np.array([0.01, 0.02])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    try:
        efficient_frontier(mu, cov, solver_options=[1])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
    try:
        key_portfolios(mu, cov, solver_options=[1])
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
