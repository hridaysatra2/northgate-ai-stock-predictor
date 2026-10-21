"""Integration contract tests (Task 4F).

All fixtures are synthetic. No real test-set performance is used to
make any recommendation; the end-to-end fixture exercises the software
contract only and is not investment performance.
"""

import copy
import math

import pandas as pd
import pytest

from src import integration as integ
from src import recommendation as rec


def forecast_record(**over):
    base = {"ticker": "TEST", "as_of_date": "2026-08-28",
            "current_price": 200.0, "predicted_log_return": 0.02,
            "model_name": "synthetic_demo"}
    base.update(over)
    return base


def risk_record(**over):
    base = {"ticker": "TEST", "as_of_date": "2026-08-28",
            "annualized_volatility": 0.15}
    base.update(over)
    return base


def weights_frame():
    rows = []
    for d in ("2020-01-02", "2020-02-03"):
        for s in ("equal_weight", "rolling_min_variance"):
            for a, w in (("AAA", 0.5), ("BBB", 0.5)):
                rows.append({"rebalance_date": d, "estimation_start":
                             "2019-01-01", "estimation_end": "2019-12-31",
                             "strategy": s, "asset": a, "weight": w,
                             "status": "ok"})
    return pd.DataFrame(rows)


# 1-3. forecast schema ------------------------------------------------
def test_1_forecast_schema_validation():
    out = integ.validate_forecast_input(forecast_record())
    assert out.ticker == "TEST" and out.as_of_date == "2026-08-28"
    assert out.current_price == pytest.approx(200.0)
    assert out.model_name == "synthetic_demo"


def test_2_missing_forecast_field_rejection():
    bad = forecast_record()
    del bad["predicted_log_return"]
    with pytest.raises(integ.MissingDependencyError):
        integ.validate_forecast_input(bad)


def test_3_invalid_forecast_date_rejection():
    with pytest.raises(ValueError):
        integ.validate_forecast_input(forecast_record(as_of_date="32-99-99"))


# 4-5. risk schema ----------------------------------------------------
def test_4_risk_schema_validation():
    out = integ.validate_risk_input(risk_record())
    assert out.annualized_volatility == pytest.approx(0.15)


def test_5_invalid_volatility_rejection():
    for bad in (float("nan"), float("inf"), -0.1, "high"):
        with pytest.raises((ValueError, TypeError)):
            integ.validate_risk_input(risk_record(annualized_volatility=bad))


# 6-9. recommendation / timestamps / horizon / conventions ------------
def test_6_recommendation_schema_compatibility():
    universe = {"TEST": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}
    forecast = rec.ForecastInput(
        ticker="TEST", as_of_date="2026-08-28", current_price=200.0,
        predicted_log_return=0.02, model_name="synthetic_demo",
        annualized_volatility=0.10)
    out = rec.recommend_one(forecast, universe).to_dict()
    checked = integ.validate_recommendation_output(out)
    assert checked["recommendation"] == "BUY"
    assert checked["risk_class"] == "normal"


def test_7_timestamp_alignment():
    ok = integ.validate_timestamp_alignment("2020-02-03", "2020-01-31")
    assert ok == {"as_of_date": "2020-02-03", "estimation_end": "2020-01-31"}
    with pytest.raises(ValueError):
        integ.validate_timestamp_alignment("2020-01-31", "2020-01-31")
    with pytest.raises(ValueError):
        integ.validate_timestamp_alignment("2020-01-15", "2020-01-31")


def test_8_prediction_horizon_consistency():
    ok = integ.validate_horizon_consistency("2024-07-29", "2024-07-29",
                                            "2024-07-30")
    assert ok["horizon_start"] == "2024-07-29"
    with pytest.raises(ValueError):
        integ.validate_horizon_consistency("2024-07-29", "2024-07-30",
                                           "2024-07-31")
    with pytest.raises(ValueError):
        integ.validate_horizon_consistency("2024-07-29", "2024-07-29",
                                           "2024-07-29")


def test_9_return_convention_compatibility():
    for convention in ("log", "simple", "price"):
        assert integ.validate_return_convention(convention) == convention
    with pytest.raises(ValueError):
        integ.validate_return_convention("adjusted")
    assert integ.check_conversion_consistency(
        0.02, math.expm1(0.02), 200.0, 200.0 * math.exp(0.02)) == \
        {"consistent": True}
    with pytest.raises(ValueError):
        integ.check_conversion_consistency(0.02, 0.05, 200.0, 204.0)


# 10-12. sentiment / synthetic / provenance ---------------------------
def test_10_sentiment_unavailable_compatibility():
    rec_out = {"sentiment_available": False, "sentiment_score": None,
               "sentiment_status": "unavailable"}
    assert integ.validate_sentiment_unavailable(rec_out) == \
        {"sentiment_available": False, "synthetic": False}
    with pytest.raises(ValueError):
        integ.validate_sentiment_unavailable(
            {"sentiment_available": True, "sentiment_score": 0.5,
             "sentiment_status": "available"})
    with pytest.raises(ValueError):
        integ.validate_sentiment_unavailable(
            {"sentiment_available": False, "sentiment_score": 0.0,
             "sentiment_status": "unavailable"})


def test_11_synthetic_demo_identified_as_synthetic():
    assert integ.is_synthetic_demo({"ticker": "DEMO_BUY",
                                    "model_name": "anything"})
    assert integ.is_synthetic_demo({"ticker": "AAPL",
                                    "model_name": "synthetic_demo"})
    assert not integ.is_synthetic_demo({"ticker": "AAPL",
                                        "model_name": "ridge"})
    assert not integ.is_synthetic_demo({})


def test_12_test_artifacts_cannot_become_live_forecasts():
    assert integ.validate_forecast_provenance("explicit_caller") == \
        "explicit_caller"
    assert integ.validate_forecast_provenance("synthetic_demo") == \
        "synthetic_demo"
    for bad in ("best_model_by_ticker", "test_metrics", "evaluation",
                "dl_best_models"):
        with pytest.raises(ValueError):
            integ.validate_forecast_provenance(bad)


# 13-14. portfolio outputs --------------------------------------------
def test_13_portfolio_output_schema_validation():
    values = pd.DataFrame({"date": ["2020-01-02"],
                           "equal_weight": [100000.0]})
    metrics = pd.DataFrame([{"strategy": "equal_weight",
                             "sharpe_ratio": 1.0}])
    summary = integ.validate_portfolio_outputs(
        weights_frame(), values, metrics, ["AAA", "BBB"])
    assert summary["benchmark_isolated"] is True
    assert summary["n_rebalance_dates"] == 2
    with pytest.raises(integ.MissingDependencyError):
        integ.validate_portfolio_outputs(
            weights_frame().drop(columns=["weight"]), values, metrics,
            ["AAA", "BBB"])


def test_14_gspc_not_a_constituent():
    bad = pd.DataFrame([{"rebalance_date": "2020-01-02",
                         "strategy": "equal_weight", "asset": "GSPC",
                         "weight": 0.5, "status": "ok"}])
    with pytest.raises(ValueError, match="benchmark"):
        integ.validate_portfolio_outputs(bad, pd.DataFrame({"date": []}),
                                         pd.DataFrame([{"strategy": "x"}]),
                                         ["AAA", "GSPC"])


# 15-17. purity guarantees --------------------------------------------
def test_15_no_mutation_of_upstream_artifacts():
    fore, risk = forecast_record(), risk_record()
    fore_copy, risk_copy = copy.deepcopy(fore), copy.deepcopy(risk)
    frame = weights_frame()
    frame_copy = frame.copy(deep=True)
    integ.validate_forecast_input(fore)
    integ.validate_risk_input(risk)
    integ.validate_portfolio_outputs(
        frame, pd.DataFrame({"date": ["2020-01-02"]}),
        pd.DataFrame([{"strategy": "equal_weight"}]), ["AAA", "BBB"])
    assert fore == fore_copy and risk == risk_copy
    pd.testing.assert_frame_equal(frame, frame_copy)


def test_16_deterministic_validation():
    first = integ.validate_forecast_input(forecast_record())
    for _ in range(5):
        assert integ.validate_forecast_input(forecast_record()) == first
    assert integ.validate_return_convention("log") == \
        integ.validate_return_convention("log")


def test_17_no_nan_inf_silent_propagation():
    with pytest.raises((ValueError, TypeError)):
        integ.validate_forecast_input(
            forecast_record(predicted_log_return=float("nan")))
    with pytest.raises((ValueError, TypeError)):
        integ.validate_forecast_input(
            forecast_record(current_price=float("inf")))
    broken = {"ticker": "T", "as_of_date": "2026-08-28",
              "current_price": 200.0, "predicted_log_return": 0.02,
              "predicted_simple_return": float("nan"),
              "predicted_price": 204.04, "model_name": "m",
              "annualized_volatility": 0.1, "risk_percentile": 0.2,
              "risk_class": "normal", "sharpe_ratio": None,
              "sortino_ratio": None, "maximum_drawdown": None,
              "historical_var": None, "sentiment_available": False,
              "sentiment_score": None, "sentiment_status": "unavailable",
              "recommendation": "BUY", "reason": "r"}
    with pytest.raises((ValueError, TypeError)):
        integ.validate_recommendation_output(broken)


# 18-19. explicit dependencies ----------------------------------------
def test_18_missing_risk_data_explicit_failure():
    with pytest.raises(integ.MissingDependencyError, match="never estimated"):
        integ.validate_risk_input({"ticker": "T",
                                   "as_of_date": "2026-08-28"})


def test_19_missing_model_selection_explicit_dependency():
    with pytest.raises(integ.MissingDependencyError,
                       match="explicit caller/configuration dependency"):
        integ.require_explicit_model_selection(None)
    with pytest.raises(integ.MissingDependencyError):
        integ.require_explicit_model_selection("")
    assert integ.require_explicit_model_selection("ridge") == "ridge"


# 20. end-to-end synthetic fixture ------------------------------------
def test_20_end_to_end_synthetic_fixture():
    """Software integration only: 200 @ 0.02 -> valid BUY, not performance."""
    fore = integ.validate_forecast_input(forecast_record())
    integ.require_explicit_model_selection(fore.model_name)
    integ.validate_forecast_provenance("synthetic_demo")
    risk = integ.validate_risk_input(risk_record())
    assert integ.is_synthetic_demo({"ticker": "TEST",
                                    "model_name": fore.model_name})
    universe = {"TEST": risk.annualized_volatility, "B": 0.20,
                "C": 0.30, "D": 0.40}
    forecast = rec.ForecastInput(
        ticker=fore.ticker, as_of_date=fore.as_of_date,
        current_price=fore.current_price,
        predicted_log_return=fore.predicted_log_return,
        model_name=fore.model_name,
        annualized_volatility=risk.annualized_volatility)
    out = rec.recommend_one(forecast, universe).to_dict()
    checked = integ.validate_recommendation_output(out)
    assert checked["recommendation"] == "BUY"
    assert checked["predicted_price"] == pytest.approx(204.04013)
    integ.validate_sentiment_unavailable(checked)
    integ.validate_timestamp_alignment(fore.as_of_date, "2026-08-27")
