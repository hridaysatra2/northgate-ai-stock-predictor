"""Research-integrity tests for the dashboard (Task 5B).

Pure helper/function tests — no browser automation. Complements (never
weakens) tests/test_dashboard.py.
"""

import hashlib
import importlib
import os

import pandas as pd
import pytest

import src.dashboard_data as dd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dashboard_sources():
    return (
        open(os.path.join(REPO, "app.py"), encoding="utf-8").read()
        + "\n"
        + open(os.path.join(REPO, "src", "dashboard_data.py"), encoding="utf-8").read()
    )


# 1. research cutoff enforced
def test_cutoff_clips_future_rows():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-08-28", "2026-08-31", "2026-09-02", "2026-10-01"]),
                       "v": [1.0, 2.0, 3.0, 4.0]})
    out = dd.clip_to_cutoff(df)
    assert out["date"].max() <= pd.to_datetime("2026-08-31")
    assert len(out) == 2
    # input not mutated
    assert len(df) == 4


def test_loaded_price_frames_respect_cutoff():
    for t in dd.TICKERS + [dd.BENCHMARK_TICKER, dd.RISK_REGIME_TICKER]:
        raw = dd.load_raw_prices(t)
        if raw is not None and "date" in raw.columns and len(raw):
            assert pd.to_datetime(raw["date"]).max() <= pd.to_datetime(dd.RESEARCH_CUTOFF)
        feat = dd.load_feature_frame(t) if t in dd.TICKERS else None
        if feat is not None and "date" in feat.columns and len(feat):
            assert pd.to_datetime(feat["date"]).max() <= pd.to_datetime(dd.RESEARCH_CUTOFF)


# 2. ADF artifact loads if available; documented reason if not
def test_adf_loads_if_available(tmp_path, monkeypatch):
    fake = tmp_path / "adf_results.csv"
    fake.write_text("ticker,statistic,p_value\nAAPL,-2.5,0.12\n", encoding="utf-8")
    monkeypatch.setitem(dd.ARTIFACT_PATHS, "adf_results", os.path.relpath(str(fake), dd.REPO_ROOT))
    df, reason = dd.load_adf_results()
    assert df is not None and len(df) == 1 and reason == "Available"


def test_adf_unavailable_reason_documented():
    df, reason = dd.load_adf_results()
    assert df is None
    assert "data/processed/eda/tables/adf_results.csv" in reason
    assert "unit-root null" in reason
    assert "95% probability" not in reason


# 3. directional accuracy formatting (proportion, never %-point error)
def test_directional_accuracy_formatting():
    assert dd.format_directional_accuracy(0.509) == "50.9%"
    assert dd.format_directional_accuracy(0.5) == "50.0%"
    src = _dashboard_sources()
    assert "proportion" in src
    assert "coin flip" in src
    # the phrase may only appear inside an explicit negation, never as endorsement
    for line in src.lower().splitlines():
        if "best for trading" in line:
            assert "no model is" in line or "not" in line, line


# 4. log/simple return labels kept distinct
def test_log_simple_return_labels():
    src = _dashboard_sources()
    assert "log return" in src.lower()
    assert "simple return" in src.lower()
    demo = dd.load_csv_artifact("demo_recommendations")
    if demo is None:
        pytest.skip("demo artifact absent")
    assert "predicted_log_return" in demo.columns
    assert "predicted_simple_return" in demo.columns
    import math
    for _, r in demo.iterrows():
        assert math.isclose(r["predicted_simple_return"], math.expm1(r["predicted_log_return"]), rel_tol=1e-9)


# 5. historical backtest labeling
def test_historical_backtest_labeling():
    assert "historical rolling backtest" in dd.HISTORICAL_BACKTEST_LABEL.lower()
    src = _dashboard_sources().lower()
    assert "historical rolling backtest" in src
    assert "not a forecast" in src
    for banned in ["expected performance", "predicted returns", "future returns"]:
        # banned as a positive claim; negations ("do not imply future returns")
        # and column names ("expected_return") are the required behavior
        for line in src.splitlines():
            if banned in line:
                assert ("not" in line or "do not" in line or "never" in line
                        or "expected_return" in line), line


# 6. synthetic recommendation isolation
def test_synthetic_isolation():
    demo = dd.load_csv_artifact("demo_recommendations")
    if demo is None:
        pytest.skip("demo artifact absent")
    assert (demo["ticker"].astype(str).str.startswith("DEMO_")).all()
    assert (demo["model_name"].astype(str) == "synthetic_demo").all()
    assert len(dd.filter_real_recommendations(demo)) == 0
    assert dd.SYNTHETIC_DEMO_LABEL == "SYNTHETIC DEMO — NOT A REAL RECOMMENDATION"
    app_src = open(os.path.join(REPO, "app.py"), encoding="utf-8").read()
    assert "SYNTHETIC_DEMO_LABEL" in app_src  # prominently used on the reco page


# 7. live recommendation remains unavailable
def test_live_recommendation_unavailable():
    from src.integration import MissingDependencyError
    with pytest.raises(MissingDependencyError):
        dd.try_generate_live_recommendation()
    st_text = dd.get_live_recommendation_status()
    assert st_text["status"] == "Live recommendations unavailable"
    assert not any([st_text["live_forecast_connected"], st_text["live_price_feed_connected"],
                    st_text["live_risk_feed_connected"], st_text["sentiment_available"]])


# 8. best_model_by_ticker cannot become live model selection
def test_best_by_ticker_cannot_become_live_selection():
    from src.integration import require_explicit_model_selection, MissingDependencyError
    with pytest.raises(MissingDependencyError):
        require_explicit_model_selection("")
    with pytest.raises(MissingDependencyError):
        require_explicit_model_selection(None)
    app_src = open(os.path.join(REPO, "app.py"), encoding="utf-8").read()
    reco_fn = app_src.split("def render_recommendations")[1].split("def render_methodology")[0]
    assert "best_by_ticker" not in reco_fn
    assert "best_model_by_ticker" not in reco_fn


# 9. benchmark not treated as portfolio constituent
def test_benchmark_not_constituent():
    for col in ["strategy_metrics", "key_portfolios", "rebalance_weights"]:
        df = dd.load_csv_artifact(col)
        if df is None:
            continue
        blob = " ".join(df.columns.astype(str)).upper()
        if col == "rebalance_weights":
            assets = set(df["asset"].astype(str).str.upper().unique())
            assert not (assets & {"GSPC", "^GSPC", "SP500", "VIX"}), assets
        else:
            for bad in ["GSPC", "^GSPC", "SP500"]:
                assert bad not in df.columns.astype(str).str.upper().tolist()
        _ = blob


# 10. no fabricated fallback data
def test_no_fabricated_fallback():
    assert dd.load_csv_artifact("no_such_key") is None
    assert dd.load_json_artifact("no_such_key") is None
    assert dd.load_text_artifact("no_such_key") is None
    assert dd.load_raw_prices("NO_SUCH_TICKER_XYZ") is None
    assert dd.artifact_exists("no_such_key") is False


# 11. dashboard does not train models
def test_dashboard_does_not_train():
    for rel in ["app.py", os.path.join("src", "dashboard_data.py")]:
        src = open(os.path.join(REPO, rel), encoding="utf-8").read()
        for banned in ["from src.models import", "from src.deep_learning import",
                       "import src.models", "import src.deep_learning",
                       ".fit(", "model.fit", "to_csv(", "to_parquet(", "torch.save"]:
            assert banned not in src, f"{rel}: {banned!r}"


# 12. dashboard does not mutate research artifacts
def test_dashboard_loads_do_not_mutate_files():
    targets = [
        "data/processed/evaluation/evaluation_metrics.csv",
        "data/processed/portfolio_frontier/key_portfolios.csv",
        "data/processed/portfolio_backtest/strategy_metrics.csv",
        "data/processed/recommendations/demo_recommendations_SYNTHETIC.csv",
    ]
    before = {t: hashlib.sha256(open(os.path.join(REPO, t), "rb").read()).hexdigest() for t in targets}
    for key in ["evaluation_metrics", "aggregate_summary", "key_portfolios", "strategy_metrics",
                "daily_values", "demo_recommendations", "dl_metrics"]:
        df = dd.load_csv_artifact(key)
        if df is not None:
            with pd.option_context("future.no_silent_downcasting", True):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df.iloc[:, :] = 0  # mutate the returned copy aggressively
    for t in dd.TICKERS[:3]:
        for loader in (dd.load_raw_prices, dd.load_feature_frame):
            _ = loader(t)
    after = {t: hashlib.sha256(open(os.path.join(REPO, t), "rb").read()).hexdigest() for t in targets}
    assert before == after
