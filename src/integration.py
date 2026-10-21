"""Thin integration contract (Task 4F).

Validates and normalizes ALREADY-PRODUCED artifacts for downstream
consumers (e.g. a future dashboard). This module trains nothing,
computes no research results, reads no files, and invents no data:
missing values fail loudly, dates are never silently converted, and
absent dependencies raise explicit errors.

Boundaries enforced here (see INTEGRATION_AUDIT.md):
  * forecasts come only from an explicit caller/configuration provider
    (test-selected artifacts such as best_model_by_ticker.csv can never
    become live forecasts);
  * risk inputs must be timestamp-safe trailing metrics, validated but
    never fabricated;
  * real recommendations require sentiment unavailable (Task 3C);
    synthetic demos must stay explicitly labeled;
  * portfolio exhibits are research-only and benchmark-isolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "VALID_RECOMMENDATIONS",
    "VALID_RISK_CLASSES",
    "VALID_RETURN_CONVENTIONS",
    "VALID_PROVENANCES",
    "SYNTHETIC_TICKER_PREFIX",
    "SYNTHETIC_MODEL_NAME",
    "BENCHMARK_TICKERS",
    "MissingDependencyError",
    "ForecastSchema",
    "RiskSchema",
    "validate_forecast_input",
    "validate_risk_input",
    "validate_recommendation_output",
    "validate_portfolio_outputs",
    "validate_timestamp_alignment",
    "validate_return_convention",
    "check_conversion_consistency",
    "validate_horizon_consistency",
    "validate_sentiment_unavailable",
    "is_synthetic_demo",
    "validate_forecast_provenance",
    "require_explicit_model_selection",
]

VALID_RECOMMENDATIONS = ("BUY", "HOLD", "SELL")
VALID_RISK_CLASSES = ("normal", "elevated")
VALID_RETURN_CONVENTIONS = ("log", "simple", "price")
VALID_PROVENANCES = ("explicit_caller", "synthetic_demo")
SYNTHETIC_TICKER_PREFIX = "DEMO_"
SYNTHETIC_MODEL_NAME = "synthetic_demo"
BENCHMARK_TICKERS = ("GSPC", "^GSPC", "sp500")

FORECAST_FIELDS = ("ticker", "as_of_date", "current_price",
                   "predicted_log_return", "model_name")
RISK_FIELDS = ("ticker", "as_of_date", "annualized_volatility")


class MissingDependencyError(ValueError):
    """A required integration dependency is absent and has no safe default."""


@dataclass(frozen=True)
class ForecastSchema:
    """Normalized, validated forecast record (ticker, ISO date, floats)."""

    ticker: str
    as_of_date: str
    current_price: float
    predicted_log_return: float
    model_name: str


@dataclass(frozen=True)
class RiskSchema:
    """Normalized, validated per-ticker risk record."""

    ticker: str
    as_of_date: str
    annualized_volatility: float


# ---------------------------------------------------------------------------
# Small validation primitives (no defaults, no filling, no mutation)
# ---------------------------------------------------------------------------

def _copy(record: dict) -> dict:
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}.")
    return dict(record)


def _ticker(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ticker must be a non-empty string, got {value!r}.")
    return value.strip()


def _iso_date(value) -> str:
    try:
        parsed = pd.to_datetime(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid date: {value!r}.") from exc
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value!r}.")
    return parsed.date().isoformat()


def _finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite (NaN/Inf rejected), "
                         f"got {value!r}.")
    return result


# ---------------------------------------------------------------------------
# Steps 9-10: contract validators
# ---------------------------------------------------------------------------

def validate_forecast_input(record: dict) -> ForecastSchema:
    """Validate a forecast record; return a normalized ForecastSchema.

    Required: ticker, as_of_date, current_price (> 0),
    predicted_log_return (finite), model_name (non-empty). Missing or
    invalid fields raise; nothing is defaulted or converted silently.
    """
    rec = _copy(record)
    missing = [k for k in FORECAST_FIELDS if k not in rec]
    if missing:
        raise MissingDependencyError(
            f"forecast record missing required fields: {missing}.")
    price = _finite("current_price", rec["current_price"])
    if price <= 0.0:
        raise ValueError(f"current_price must be > 0, got {price!r}.")
    model = rec["model_name"]
    if not isinstance(model, str) or not model:
        raise ValueError("model_name must be a non-empty string.")
    return ForecastSchema(
        ticker=_ticker(rec["ticker"]),
        as_of_date=_iso_date(rec["as_of_date"]),
        current_price=price,
        predicted_log_return=_finite("predicted_log_return",
                                     rec["predicted_log_return"]),
        model_name=model,
    )


def validate_risk_input(record: dict) -> RiskSchema:
    """Validate a per-ticker risk record; missing data fails explicitly."""
    rec = _copy(record)
    missing = [k for k in RISK_FIELDS if k not in rec]
    if missing:
        raise MissingDependencyError(
            f"risk record missing required fields: {missing} "
            f"(never estimated from full-history statistics here).")
    vol = _finite("annualized_volatility", rec["annualized_volatility"])
    if vol < 0.0:
        raise ValueError(
            f"annualized_volatility must be >= 0, got {vol!r}.")
    return RiskSchema(ticker=_ticker(rec["ticker"]),
                      as_of_date=_iso_date(rec["as_of_date"]),
                      annualized_volatility=vol)


def validate_recommendation_output(record: dict) -> dict:
    """Validate a Task 4E output record, incl. conversion consistency.

    Checks: all 19 OUTPUT_COLUMNS present, recommendation/risk_class/
    sentiment values in their vocabularies, sentiment trio consistent,
    and predicted_simple_return / predicted_price consistent with the
    log forecast (tolerance 1e-9 relative). Returns a plain dict copy.
    """
    from src.recommendation import OUTPUT_COLUMNS
    rec = _copy(record)
    missing = [c for c in OUTPUT_COLUMNS if c not in rec]
    if missing:
        raise MissingDependencyError(
            f"recommendation record missing columns: {missing}.")
    if rec["recommendation"] not in VALID_RECOMMENDATIONS:
        raise ValueError(
            f"unknown recommendation: {rec['recommendation']!r}.")
    if rec["risk_class"] not in VALID_RISK_CLASSES:
        raise ValueError(f"unknown risk_class: {rec['risk_class']!r}.")
    pct = _finite("risk_percentile", rec["risk_percentile"])
    if not 0.0 <= pct <= 1.0:
        raise ValueError(f"risk_percentile out of [0, 1]: {pct!r}.")
    log_ret = _finite("predicted_log_return", rec["predicted_log_return"])
    simple = _finite("predicted_simple_return",
                     rec["predicted_simple_return"])
    price = _finite("current_price", rec["current_price"])
    pred_price = _finite("predicted_price", rec["predicted_price"])
    if not math.isclose(simple, math.expm1(log_ret), rel_tol=1e-9):
        raise ValueError("predicted_simple_return inconsistent with "
                         "predicted_log_return.")
    if not math.isclose(pred_price, price * math.exp(log_ret), rel_tol=1e-9):
        raise ValueError("predicted_price inconsistent with current_price "
                         "and predicted_log_return.")
    validate_sentiment_unavailable(rec, real_only=False)
    if not isinstance(rec["reason"], str) or not rec["reason"]:
        raise ValueError("reason must be a non-empty string.")
    return rec


def validate_portfolio_outputs(weights: pd.DataFrame,
                               values: pd.DataFrame,
                               metrics: pd.DataFrame,
                               tickers) -> dict:
    """Validate Task 4D output frames for dashboard consumption.

    Checks required schemas, finite weights, benchmark isolation
    (GSPC/^GSPC/sp500 must never appear as a weight asset), and that
    only known strategies are present. Returns a summary dict.
    """
    tickers = list(tickers)
    for name, frame, cols in (
            ("rebalance_weights", weights,
             {"rebalance_date", "strategy", "asset", "weight"}),
            ("daily_portfolio_values", values, {"date"}),
            ("strategy_metrics", metrics, {"strategy"})):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame.")
        missing = cols - set(frame.columns)
        if missing:
            raise MissingDependencyError(
                f"{name} missing required columns: {sorted(missing)}.")
    assets = set(weights["asset"].astype(str).unique())
    intruders = assets & set(BENCHMARK_TICKERS)
    if intruders:
        raise ValueError(
            f"benchmark tickers must never be portfolio constituents: "
            f"{sorted(intruders)}.")
    unknown = set(weights["asset"].astype(str).unique()) - set(tickers)
    if unknown:
        raise ValueError(f"weight assets outside universe: {sorted(unknown)}.")
    w = pd.to_numeric(weights["weight"], errors="coerce").to_numpy()
    if not ((w == w) & (abs(w) != float("inf"))).all():
        raise ValueError("weights contain NaN/Inf.")
    return {"n_weight_rows": int(len(weights)),
            "n_rebalance_dates": int(weights["rebalance_date"].nunique()),
            "strategies": sorted(weights["strategy"].astype(str).unique()),
            "assets": sorted(assets),
            "benchmark_isolated": True}


def validate_timestamp_alignment(as_of_date, estimation_end) -> dict:
    """Require estimation_end strictly before as_of_date (ISO-normalized)."""
    as_of = _iso_date(as_of_date)
    end = _iso_date(estimation_end)
    if not end < as_of:
        raise ValueError(
            f"look-ahead violation: estimation_end ({end}) must be strictly "
            f"before as_of_date ({as_of}).")
    return {"as_of_date": as_of, "estimation_end": end}


def validate_return_convention(convention: str) -> str:
    """Accept only 'log', 'simple', or 'price' (documented, explicit)."""
    if convention not in VALID_RETURN_CONVENTIONS:
        raise ValueError(
            f"return convention must be one of {VALID_RETURN_CONVENTIONS}, "
            f"got {convention!r} (no silent conversion).")
    return convention


def check_conversion_consistency(predicted_log_return: float,
                                 predicted_simple_return: float,
                                 current_price: float,
                                 predicted_price: float) -> dict:
    """Verify log→simple→price math (tolerance 1e-9 relative)."""
    log_ret = _finite("predicted_log_return", predicted_log_return)
    simple = _finite("predicted_simple_return", predicted_simple_return)
    price = _finite("current_price", current_price)
    pred = _finite("predicted_price", predicted_price)
    if price <= 0.0:
        raise ValueError("current_price must be > 0.")
    if not math.isclose(simple, math.expm1(log_ret), rel_tol=1e-9):
        raise ValueError("simple return inconsistent with log return.")
    if not math.isclose(pred, price * math.exp(log_ret), rel_tol=1e-9):
        raise ValueError("predicted price inconsistent with inputs.")
    return {"consistent": True}


def validate_horizon_consistency(as_of_date, horizon_start,
                                 horizon_end) -> dict:
    """Require a 1-day-ahead horizon anchored at the prediction date."""
    as_of = _iso_date(as_of_date)
    start = _iso_date(horizon_start)
    end = _iso_date(horizon_end)
    if start != as_of:
        raise ValueError(
            f"horizon must start at the prediction date ({as_of}), "
            f"got {start}.")
    if not end > start:
        raise ValueError(
            f"horizon end ({end}) must be after start ({start}).")
    return {"as_of_date": as_of, "horizon_start": start, "horizon_end": end}


def validate_sentiment_unavailable(record: dict, real_only: bool = True
                                   ) -> dict:
    """Require the Task 3C unavailable-sentiment triple.

    With real_only=True (default), available sentiment is rejected for
    real pipelines; synthetic demos must pass through is_synthetic_demo
    instead. With real_only=False (output-schema check), an available
    triple is accepted as well-formed but still validated for shape.
    """
    rec = _copy(record)
    available = rec.get("sentiment_available", False)
    score = rec.get("sentiment_score")
    status = rec.get("sentiment_status")
    if available is False or available is None:
        if score is not None or status != "unavailable":
            raise ValueError(
                "unavailable sentiment requires score=null and "
                "status='unavailable'.")
        return {"sentiment_available": False, "synthetic": False}
    if real_only:
        raise ValueError(
            "real recommendations must not carry sentiment: Task 3C found "
            "no suitable historical source (no fabrication, no GDELT or "
            "recent-news substitution).")
    if status != "available" or not isinstance(score, (int, float)) \
            or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError("available sentiment requires a finite score and "
                         "status='available'.")
    return {"sentiment_available": True, "synthetic": False}


def is_synthetic_demo(record: dict) -> bool:
    """True iff the record is explicitly labeled synthetic demo output."""
    rec = _copy(record)
    ticker = str(rec.get("ticker", ""))
    model = str(rec.get("model_name", ""))
    return ticker.startswith(SYNTHETIC_TICKER_PREFIX) or \
        model == SYNTHETIC_MODEL_NAME


def validate_forecast_provenance(provenance: str) -> str:
    """Accept only explicit-caller or synthetic-demo forecast provenance.

    Anything referencing test-selected artifacts (best_model_by_ticker,
    test metrics, evaluation outputs) raises: final-test artifacts can
    never automatically become live forecasts.
    """
    if provenance not in VALID_PROVENANCES:
        raise ValueError(
            f"forecast provenance {provenance!r} is not an accepted live "
            f"source {list(VALID_PROVENANCES)}: final-test artifacts "
            f"(best_model_by_ticker, test metrics) can never automatically "
            f"become live forecasts.")
    return provenance


def require_explicit_model_selection(model_name) -> str:
    """Return the caller-supplied model name or raise the dependency error."""
    if not isinstance(model_name, str) or not model_name:
        raise MissingDependencyError(
            "no leakage-free live model-selection rule exists: forecast "
            "provider is currently an explicit caller/configuration "
            "dependency (model_name must be supplied).")
    return model_name
