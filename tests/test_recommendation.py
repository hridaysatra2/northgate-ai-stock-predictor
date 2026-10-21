"""Tests for src.recommendation (Task 4E).

Covers all five rule branches, conversions, boundaries, validation,
determinism, sentiment handling, and a leakage audit proving the engine
never reads test outcomes or repository model artifacts.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from src import recommendation as rec


def make_input(ticker="AAA", price=100.0, log_return=0.02, vol=0.15,
               model="test_model", date="2026-08-28", **kwargs):
    return rec.ForecastInput(
        ticker=ticker, as_of_date=date, current_price=price,
        predicted_log_return=log_return, model_name=model,
        annualized_volatility=vol, **kwargs)


def universe(*vols, tickers=("A", "B", "C", "D")):
    assert len(vols) == len(tickers)
    return dict(zip(tickers, vols))


# ---------------------------------------------------------------------------
# 1-5. The five rule branches
# ---------------------------------------------------------------------------

def test_strong_positive_normal_risk_gives_buy():
    out = rec.recommend_one(make_input(), universe(0.15, 0.20, 0.25, 0.30)
                            | {"AAA": 0.10})
    assert out.recommendation == "BUY"
    assert out.risk_class == "normal"
    assert out.reason.startswith(
        "Forecast return is strongly positive and current volatility "
        "is within the normal risk range.")


def test_strong_positive_elevated_risk_gives_hold():
    out = rec.recommend_one(make_input(vol=0.90),
                            {"AAA": 0.90, "B": 0.10, "C": 0.15, "D": 0.20})
    assert out.recommendation == "HOLD"
    assert out.risk_class == "elevated"
    assert "but current volatility is elevated" in out.reason


def test_neutral_forecast_gives_hold():
    out = rec.recommend_one(make_input(log_return=0.002),
                            {"AAA": 0.15, "B": 0.15, "C": 0.15, "D": 0.15})
    assert out.recommendation == "HOLD"
    assert out.reason.startswith(
        "Forecast return is within the neutral range.")


def test_negative_normal_risk_gives_hold():
    out = rec.recommend_one(make_input(log_return=-0.02, vol=0.10),
                            {"AAA": 0.10, "B": 0.20, "C": 0.30, "D": 0.40})
    assert out.recommendation == "HOLD"
    assert "not elevated" in out.reason


def test_negative_elevated_risk_gives_sell():
    out = rec.recommend_one(make_input(log_return=-0.03, vol=0.90),
                            {"AAA": 0.90, "B": 0.10, "C": 0.20, "D": 0.30})
    assert out.recommendation == "SELL"
    assert out.reason.startswith(
        "Forecast return is negative and current volatility is elevated.")


# ---------------------------------------------------------------------------
# 6-7. Forecast conversions (spec example: 200 @ 0.02 -> 2.0201%, 204.04)
# ---------------------------------------------------------------------------

def test_log_to_simple_conversion():
    assert rec.log_to_simple_return(0.02) == pytest.approx(0.02020134)
    assert rec.log_to_simple_return(0.0) == pytest.approx(0.0)
    assert rec.log_to_simple_return(-0.03) == pytest.approx(-0.02955447)


def test_predicted_price_conversion():
    assert rec.predict_price(200.0, 0.02) == pytest.approx(204.04013)
    out = rec.recommend_one(make_input(price=200.0, log_return=0.02),
                            {"AAA": 0.10})
    assert out.predicted_simple_return == pytest.approx(0.02020134)
    assert out.predicted_price == pytest.approx(204.04013)


# ---------------------------------------------------------------------------
# 8-10. Boundaries
# ---------------------------------------------------------------------------

def test_threshold_boundaries_inclusive():
    thr = rec.default_thresholds()
    assert rec.classify_forecast(0.01, thr) == "strong_positive"
    assert rec.classify_forecast(-0.01, thr) == "negative"
    assert rec.classify_forecast(0.009999, thr) == "neutral"
    assert rec.classify_forecast(-0.009999, thr) == "neutral"
    assert rec.classify_risk(0.75, thr) == "elevated"
    assert rec.classify_risk(0.749999, thr) == "normal"


def test_volatility_percentile_counting_definition():
    assert rec.volatility_percentile(0.10, [0.1, 0.2, 0.3, 0.4]) == \
        pytest.approx(0.25)
    assert rec.volatility_percentile(0.30, [0.1, 0.2, 0.3, 0.4]) == \
        pytest.approx(0.75)
    assert rec.volatility_percentile(0.40, [0.1, 0.2, 0.3, 0.4]) == \
        pytest.approx(1.0)
    # ties share one value; single-asset universe yields 1.0
    assert rec.volatility_percentile(0.20, [0.2, 0.2, 0.2]) == \
        pytest.approx(1.0)
    assert rec.volatility_percentile(0.05, [0.05]) == pytest.approx(1.0)


def test_75th_percentile_elevated_boundary_end_to_end():
    # vols -> percentiles 0.25/0.5/0.75/1.0: top two elevated at 0.75
    vols = universe(0.10, 0.20, 0.30, 0.40)
    classes = [rec.recommend_one(make_input(ticker=t, vol=v), vols).risk_class
               for t, v in vols.items()]
    assert classes == ["normal", "normal", "elevated", "elevated"]


# ---------------------------------------------------------------------------
# 11/21. Sentiment unavailable, never fabricated, never acted upon
# ---------------------------------------------------------------------------

def test_sentiment_unavailable_by_default():
    out = rec.recommend_one(make_input(), {"AAA": 0.10})
    assert out.sentiment_available is False
    assert out.sentiment_score is None
    assert out.sentiment_status == "unavailable"
    assert "Historical sentiment unavailable" in out.reason


def test_no_sentiment_score_fabricated():
    frame = rec.recommend_batch([make_input(ticker="A", vol=0.10),
                                 make_input(ticker="B", vol=0.20)])
    assert frame["sentiment_score"].isna().all()
    assert (frame["sentiment_status"] == "unavailable").all()


def test_sentiment_toggle_does_not_change_recommendation():
    base = make_input()
    vols = {"AAA": 0.10}
    off = rec.recommend_one(base, vols)
    on = rec.recommend_one(
        make_input(sentiment_available=True, sentiment_score=0.8), vols)
    assert off.recommendation == on.recommendation
    assert on.sentiment_status == "available"
    assert on.sentiment_score == pytest.approx(0.8)


def test_sentiment_available_requires_score():
    with pytest.raises(ValueError):
        make_input(sentiment_available=True)


# ---------------------------------------------------------------------------
# 12-17. Validation
# ---------------------------------------------------------------------------

def test_null_optionals_preserved():
    out = rec.recommend_one(
        make_input(sharpe_ratio=1.2, sortino_ratio=None,
                   maximum_drawdown=-0.1, historical_var=None),
        {"AAA": 0.10})
    assert out.sharpe_ratio == pytest.approx(1.2)
    assert out.sortino_ratio is None
    assert out.maximum_drawdown == pytest.approx(-0.1)
    assert out.historical_var is None


def test_nan_forecast_rejected():
    with pytest.raises(ValueError):
        make_input(log_return=float("nan"))


def test_infinite_forecast_rejected():
    with pytest.raises(ValueError):
        make_input(log_return=float("inf"))
    with pytest.raises(ValueError):
        rec.log_to_simple_return(float("-inf"))


def test_zero_negative_price_rejected():
    with pytest.raises(ValueError):
        make_input(price=0.0)
    with pytest.raises(ValueError):
        make_input(price=-5.0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        make_input(ticker="   ")
    with pytest.raises(ValueError):
        make_input(date="not-a-date")
    with pytest.raises(ValueError):
        make_input(vol=float("nan"))
    with pytest.raises(ValueError):
        make_input(vol=-0.1)
    with pytest.raises(ValueError):
        make_input(vol=float("inf"))
    with pytest.raises(ValueError):
        make_input(model="")
    with pytest.raises(ValueError):
        rec.volatility_percentile(0.1, [])
    with pytest.raises(ValueError):
        rec.volatility_percentile(0.1, [0.1, float("nan")])


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        rec.RecommendationThresholds(positive_return_threshold=float("nan"))
    with pytest.raises(ValueError):
        rec.RecommendationThresholds(positive_return_threshold=0.01,
                                     negative_return_threshold=0.02)
    with pytest.raises(ValueError):
        rec.RecommendationThresholds(elevated_risk_percentile=1.5)
    with pytest.raises(ValueError):
        rec.RecommendationThresholds(elevated_risk_percentile=-0.1)
    with pytest.raises(TypeError):
        rec.thresholds_from_dict("not-a-dict")


def test_invalid_percentile_rejected():
    with pytest.raises(ValueError):
        rec.classify_risk(1.5)
    with pytest.raises(ValueError):
        rec.classify_risk(-0.1)
    with pytest.raises(ValueError):
        rec.recommend_one(make_input(), {"BBB": 0.10})  # ticker absent
    with pytest.raises(ValueError):
        rec.recommend_one(make_input(), {})


# ---------------------------------------------------------------------------
# 18-20. Determinism, batches, model names
# ---------------------------------------------------------------------------

def test_deterministic_repeated_execution():
    vols = {"AAA": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}
    first = rec.recommend_one(make_input(), vols).to_dict()
    for _ in range(5):
        assert rec.recommend_one(make_input(), vols).to_dict() == first


def test_batch_multiple_tickers_uses_batch_as_universe():
    frame = rec.recommend_batch([
        make_input(ticker="A", log_return=0.03, vol=0.10),
        make_input(ticker="B", log_return=-0.03, vol=0.40),
        make_input(ticker="C", log_return=0.001, vol=0.20),
        make_input(ticker="D", log_return=0.03, vol=0.30),
    ])
    assert list(frame.columns) == rec.OUTPUT_COLUMNS
    by_ticker = frame.set_index("ticker")["recommendation"].to_dict()
    # percentiles: A .25 normal, B 1.0 elevated, C .5, D .75 elevated
    assert by_ticker == {"A": "BUY", "B": "SELL", "C": "HOLD", "D": "HOLD"}
    assert frame.set_index("ticker").loc["D", "risk_percentile"] == \
        pytest.approx(0.75)


def test_batch_from_dataframe_and_model_name_preserved():
    df = pd.DataFrame([{
        "ticker": "ZZZ", "as_of_date": "2026-08-28", "current_price": 50.0,
        "predicted_log_return": 0.05,
        "model_name": "my_custom_model_v2 (ridge, alpha=100)",
        "annualized_volatility": 0.12,
    }])
    frame = rec.recommend_batch(df)
    assert frame.loc[0, "model_name"] == \
        "my_custom_model_v2 (ridge, alpha=100)"
    # single-row batch: percentile is 1.0 -> elevated, so HOLD, not BUY
    assert frame.loc[0, "recommendation"] == "HOLD"
    with pytest.raises(ValueError):
        rec.recommend_batch(pd.DataFrame([{"ticker": "Z"}]))  # missing cols
    with pytest.raises(ValueError):
        rec.recommend_batch([])


def test_custom_thresholds_change_branches_without_scores():
    strict = rec.RecommendationThresholds(
        positive_return_threshold=0.05, negative_return_threshold=-0.05,
        elevated_risk_percentile=0.75)
    out = rec.recommend_one(make_input(log_return=0.02), {"AAA": 0.10},
                            strict)
    assert out.recommendation == "HOLD"  # 0.02 now neutral, not positive
    loaded = rec.thresholds_from_dict({
        "positive_return_threshold": 0.01,
        "negative_return_threshold": -0.01,
        "elevated_risk_percentile": 0.75})
    assert loaded == rec.default_thresholds()


# ---------------------------------------------------------------------------
# 22. Reason consistency + leakage audit
# ---------------------------------------------------------------------------

def test_reason_text_generated_consistently():
    cases = [
        (dict(log_return=0.05, vol=0.10),
         {"AAA": 0.10, "B": 0.20, "C": 0.30, "D": 0.50}, "BUY"),
        (dict(log_return=0.05, vol=0.40),
         {"AAA": 0.40, "B": 0.20, "C": 0.30, "D": 0.50}, "HOLD"),
        (dict(log_return=0.0, vol=0.10),
         {"AAA": 0.10, "B": 0.20, "C": 0.30, "D": 0.50}, "HOLD"),
        (dict(log_return=-0.05, vol=0.10),
         {"AAA": 0.10, "B": 0.20, "C": 0.30, "D": 0.50}, "HOLD"),
        (dict(log_return=-0.05, vol=0.40),
         {"AAA": 0.40, "B": 0.20, "C": 0.30, "D": 0.50}, "SELL"),
    ]
    for kwargs, vols, expected in cases:
        first = rec.recommend_one(make_input(**kwargs), vols)
        assert first.recommendation == expected
        assert rec.recommend_one(make_input(**kwargs), vols).reason == \
            first.reason
        assert first.reason.endswith(
            "Historical sentiment unavailable; recommendation uses "
            "forecast and risk only.")


def test_engine_is_pure_transformation_of_inputs():
    """Leakage audit: no reads of test outcomes or model artifacts.

    Documentation may NAME the forbidden artifacts (to record the
    prohibition); what must be absent is any executable access: file
    reads, data-library imports, or references to outcome columns.
    """
    source = Path("src/recommendation.py").read_text(encoding="utf-8")
    for forbidden in ("read_csv(", "read_parquet(", "open(", "yfinance",
                      "actual_return", "directional_accuracy",
                      "from src import", "from src.", "import src"):
        assert forbidden not in source, f"leakage risk: {forbidden!r}"
    # works with a fabricated model label: no repo artifact consulted
    out = rec.recommend_one(
        make_input(model="never_trained_fictional_model_xyz"),
        {"AAA": 0.15, "B": 0.20})
    assert out.model_name == "never_trained_fictional_model_xyz"
    assert out.recommendation == "BUY"


def test_output_schema_column_order():
    out = rec.recommend_one(make_input(), {"AAA": 0.10})
    assert list(out.to_dict()) == rec.OUTPUT_COLUMNS
