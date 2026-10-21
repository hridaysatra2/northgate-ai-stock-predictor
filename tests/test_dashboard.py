"""Dashboard tests (Task 5A) — pure helper-function tests, no browser automation."""

import importlib
import os

import pandas as pd
import pytest

import src.dashboard_data as dd


# 1. app imports successfully
def test_app_imports():
    mod = importlib.import_module("app")
    assert mod is not None


# 2. required dashboard modules/functions exist
def test_required_modules_functions_exist():
    app = importlib.import_module("app")
    for name in [
        "NAV_PAGES", "TICKERS", "RESEARCH_CUTOFF", "DISCLAIMER",
        "HISTORICAL_BACKTEST_LABEL", "SYNTHETIC_DEMO_LABEL",
        "render_overview", "render_stock_analysis", "render_model_performance",
        "render_portfolio_optimization", "render_portfolio_backtest",
        "render_recommendations", "render_methodology", "main",
    ]:
        assert hasattr(app, name), f"app missing {name}"
    for name in [
        "get_artifact_paths", "artifact_status", "load_csv_artifact",
        "is_synthetic_ticker", "is_synthetic_record",
        "filter_real_recommendations", "try_generate_live_recommendation",
        "get_live_recommendation_status", "get_research_cutoff",
    ]:
        assert hasattr(dd, name), f"dashboard_data missing {name}"


# 3. expected artifact paths are handled
def test_expected_artifact_paths_handled():
    paths = dd.get_artifact_paths()
    for key in ["evaluation_metrics", "dl_metrics", "key_portfolios",
                "efficient_frontier", "daily_values", "strategy_metrics",
                "demo_recommendations", "backtest_summary"]:
        assert key in paths
    status = dd.artifact_status()
    assert set(status) == set(paths)
    assert set(status.values()) <= {"Available", "Unavailable"}


# 4. missing optional artifact does not cause fabricated output
def test_missing_optional_artifact_no_fabrication(monkeypatch):
    monkeypatch.setitem(dd.ARTIFACT_PATHS, "_missing_test_xyz", "data/processed/does_not_exist_xyz.csv")
    try:
        assert dd.load_csv_artifact("_missing_test_xyz") is None
        assert dd.load_json_artifact("_missing_test_xyz") is None
        assert dd.load_text_artifact("_missing_test_xyz") is None
        assert dd.artifact_exists("_missing_test_xyz") is False
    finally:
        dd.ARTIFACT_PATHS.pop("_missing_test_xyz", None)


# 5. synthetic recommendation data is explicitly identified
def test_synthetic_demo_explicitly_identified():
    assert dd.SYNTHETIC_DEMO_LABEL == "SYNTHETIC DEMO — NOT A REAL RECOMMENDATION"
    assert dd.is_synthetic_ticker("DEMO_BUY") is True
    assert dd.is_synthetic_ticker("AAPL") is False
    assert dd.is_synthetic_record({"ticker": "DEMO_X", "model_name": "whatever"}) is True
    assert dd.is_synthetic_record({"ticker": "AAPL", "model_name": "synthetic_demo"}) is True
    assert dd.is_synthetic_record({"ticker": "AAPL", "model_name": "ridge"}) is False
    demo = dd.load_csv_artifact("demo_recommendations")
    if demo is not None:
        labeled = dd.label_demo_frame(demo)
        assert "demo_label" in labeled.columns
        assert (labeled["demo_label"] == dd.SYNTHETIC_DEMO_LABEL).all()


# 6. real recommendation page cannot treat DEMO_* as real tickers
def test_demo_tickers_never_real():
    demo = dd.load_csv_artifact("demo_recommendations")
    if demo is None:
        pytest.skip("demo artifact absent")
    real = dd.filter_real_recommendations(demo)
    assert len(real) == 0
    for t in demo["ticker"].astype(str).tolist():
        assert dd.is_synthetic_ticker(t)


# 7. no training functions are called by dashboard loading
def test_no_training_functions_in_dashboard():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ["app.py", os.path.join("src", "dashboard_data.py")]:
        src = open(os.path.join(repo, rel), encoding="utf-8").read()
        for banned in ["from src.models import", "from src.deep_learning import",
                       "from src.features import", "import src.models",
                       "import src.deep_learning", ".fit(", "model.fit"]:
            assert banned not in src, f"{rel} must not contain {banned!r}"


# 8. no recommendation is generated when live forecast inputs are absent
def test_no_recommendation_without_live_inputs():
    from src.integration import MissingDependencyError
    with pytest.raises(MissingDependencyError):
        dd.try_generate_live_recommendation(forecast=None, risk=None)
    with pytest.raises(MissingDependencyError):
        dd.try_generate_live_recommendation(forecast={"ticker": "AAPL"}, risk=None)
    status = dd.get_live_recommendation_status()
    assert status["status"] == dd.LIVE_RECOMMENDATIONS_UNAVAILABLE
    assert status["live_forecast_connected"] is False
    assert status["sentiment_available"] is False


# 9. portfolio artifacts load without mutation
def test_portfolio_artifacts_load_without_mutation():
    key = dd.load_csv_artifact("key_portfolios")
    strat = dd.load_csv_artifact("strategy_metrics")
    if key is None or strat is None:
        pytest.skip("portfolio artifacts absent")
    before = key.copy(deep=True)
    key["AAPL"] = -999.0  # mutate the returned copy
    again = dd.load_csv_artifact("key_portfolios")
    pd.testing.assert_frame_equal(again, before)
    assert (again["AAPL"] != -999.0).all()


# 10. historical backtest is labeled historical
def test_backtest_labeled_historical():
    assert "Historical" in dd.HISTORICAL_BACKTEST_LABEL
    app = importlib.import_module("app")
    src = open(app.__file__, encoding="utf-8").read()
    assert "HISTORICAL_BACKTEST_LABEL" in src or "Historical rolling backtest" in src
    assert "not a forecast" in src.lower() or "not currently generated" in src


# 11. research cutoff is preserved
def test_research_cutoff_preserved():
    assert dd.get_research_cutoff() == "2026-08-31"
    assert dd.RESEARCH_CUTOFF == "2026-08-31"
    app = importlib.import_module("app")
    assert app.RESEARCH_CUTOFF == "2026-08-31"


# 12. required navigation/pages exist
def test_required_navigation_pages_exist():
    assert dd.NAV_PAGES == [
        "Overview", "Stock Analysis", "Model Performance",
        "Portfolio Optimization", "Portfolio Backtest",
        "Recommendations", "Methodology & Limitations",
    ]
    app = importlib.import_module("app")
    assert app.NAV_PAGES == dd.NAV_PAGES
