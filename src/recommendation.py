"""Explainable rule-based recommendation engine (Task 4E).

Decision-support only: converts an explicitly supplied forecast plus
currently available risk information into BUY / HOLD / SELL with a
human-readable reason. Recommendations guarantee nothing and predict
nothing with certainty.

MODEL-SELECTION POSITION (leakage audit, 2026-09-27):
  * ``data/processed/evaluation/best_model_by_ticker.csv`` is derived
    from final test-set metrics (``evaluation_metrics.csv``:
    ``metric_source=predictions``, ``n_test=524``). It is therefore a
    TEST-SELECTED artifact and this engine MUST NOT use it for live
    model selection.
  * ``selected_hyperparameters.csv`` carries dev-set CV diagnostics, but
    the repository defines no live model-selection rule on top of them.
    Adopting one here (e.g. argmin dev_cv_rmse) would invent a post-hoc
    methodology, which is prohibited.
  * Consequently this engine performs NO model selection at all. The
    caller/configuration explicitly supplies every forecast as
    (ticker, as_of_date, current_price, predicted_log_return,
    model_name). ``model_name`` is an opaque label, preserved exactly
    and never validated against repository artifacts.

SENTIMENT POSITION (Task 3C audit): no suitable historical news source
exists, so ``sentiment_available`` is False by default;
``sentiment_score`` stays null, ``sentiment_status`` is "unavailable",
and sentiment never alters the recommendation.

RISK POSITION: classification uses ONLY the supplied current
annualized volatilities. For each batch timestamp, percentile(v) =
fraction of universe volatilities at or below v (counting definition);
risk is "elevated" iff percentile >= threshold (default 0.75).
No weighted score is constructed anywhere: risk only modifies the
forecast rule through the five explicit branches below.

RECOMMENDATION RULES (exact):
  strong positive (simple >= +thr) + normal risk   -> BUY
  strong positive (simple >= +thr) + elevated risk -> HOLD
  neutral (neg_thr < simple < pos_thr)             -> HOLD
  negative (simple <= neg_thr) + normal risk       -> HOLD
  negative (simple <= neg_thr) + elevated risk     -> SELL

THRESHOLDS (defaults: +0.01 / -0.01 / 0.75) are configurable defaults,
NOT empirically validated constants. They were not tuned against test
results and must not be.

The engine never reads model predictions, evaluation metrics, market
data files, or future outcomes: it is a pure transformation of its
inputs. Historical use requires the CALLER to supply timestamp-safe
inputs; the engine itself runs no backtest and claims no historical
performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "BUY",
    "HOLD",
    "SELL",
    "RISK_NORMAL",
    "RISK_ELEVATED",
    "FORECAST_STRONG_POSITIVE",
    "FORECAST_NEUTRAL",
    "FORECAST_NEGATIVE",
    "SENTIMENT_UNAVAILABLE",
    "SENTIMENT_AVAILABLE",
    "DEFAULT_POSITIVE_RETURN_THRESHOLD",
    "DEFAULT_NEGATIVE_RETURN_THRESHOLD",
    "DEFAULT_ELEVATED_RISK_PERCENTILE",
    "OUTPUT_COLUMNS",
    "RecommendationThresholds",
    "ForecastInput",
    "Recommendation",
    "default_thresholds",
    "thresholds_from_dict",
    "log_to_simple_return",
    "predict_price",
    "classify_forecast",
    "volatility_percentile",
    "classify_risk",
    "apply_recommendation_rule",
    "build_reason",
    "recommend_one",
    "recommend_batch",
    "run_demo",
    "main",
]

BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"

RISK_NORMAL = "normal"
RISK_ELEVATED = "elevated"

FORECAST_STRONG_POSITIVE = "strong_positive"
FORECAST_NEUTRAL = "neutral"
FORECAST_NEGATIVE = "negative"

SENTIMENT_UNAVAILABLE = "unavailable"
SENTIMENT_AVAILABLE = "available"

DEFAULT_POSITIVE_RETURN_THRESHOLD = 0.01
DEFAULT_NEGATIVE_RETURN_THRESHOLD = -0.01
DEFAULT_ELEVATED_RISK_PERCENTILE = 0.75

OUTPUT_COLUMNS = [
    "ticker",
    "as_of_date",
    "current_price",
    "predicted_log_return",
    "predicted_simple_return",
    "predicted_price",
    "model_name",
    "annualized_volatility",
    "risk_percentile",
    "risk_class",
    "sharpe_ratio",
    "sortino_ratio",
    "maximum_drawdown",
    "historical_var",
    "sentiment_available",
    "sentiment_score",
    "sentiment_status",
    "recommendation",
    "reason",
]

_REASON = {
    (BUY, FORECAST_STRONG_POSITIVE, RISK_NORMAL):
        "Forecast return is strongly positive and current volatility "
        "is within the normal risk range.",
    (HOLD, FORECAST_STRONG_POSITIVE, RISK_ELEVATED):
        "Forecast return is strongly positive, but current volatility "
        "is elevated.",
    (HOLD, FORECAST_NEUTRAL, RISK_NORMAL):
        "Forecast return is within the neutral range.",
    (HOLD, FORECAST_NEUTRAL, RISK_ELEVATED):
        "Forecast return is within the neutral range.",
    (HOLD, FORECAST_NEGATIVE, RISK_NORMAL):
        "Forecast return is negative, but current volatility is not "
        "elevated.",
    (SELL, FORECAST_NEGATIVE, RISK_ELEVATED):
        "Forecast return is negative and current volatility is elevated.",
}

_SENTIMENT_SUFFIX_UNAVAILABLE = (
    "Historical sentiment unavailable; recommendation uses forecast "
    "and risk only."
)
_SENTIMENT_SUFFIX_AVAILABLE = (
    "Sentiment data is available and recorded; the recommendation rule "
    "uses forecast and risk only."
)


# Configuration

@dataclass(frozen=True)
class RecommendationThresholds:
    """Decision thresholds (configurable defaults, not validated optima).

    positive_return_threshold: strong-positive iff simple return >= this.
    negative_return_threshold: negative iff simple return <= this.
    elevated_risk_percentile: elevated iff volatility percentile >= this.
    """

    positive_return_threshold: float = DEFAULT_POSITIVE_RETURN_THRESHOLD
    negative_return_threshold: float = DEFAULT_NEGATIVE_RETURN_THRESHOLD
    elevated_risk_percentile: float = DEFAULT_ELEVATED_RISK_PERCENTILE

    def __post_init__(self) -> None:
        for name in ("positive_return_threshold",
                     "negative_return_threshold",
                     "elevated_risk_percentile"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {value!r}.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")
        if not self.positive_return_threshold > self.negative_return_threshold:
            raise ValueError(
                "positive_return_threshold must exceed "
                "negative_return_threshold.")
        pct = self.elevated_risk_percentile
        if not 0.0 <= pct <= 1.0:
            raise ValueError(
                f"elevated_risk_percentile must be between 0 and 1, "
                f"got {pct!r}.")


def default_thresholds() -> RecommendationThresholds:
    """Return the default (non-validated, non-tuned) thresholds."""
    return RecommendationThresholds()


def thresholds_from_dict(values: dict) -> RecommendationThresholds:
    """Build thresholds from a mapping (e.g. ``config.yaml`` section)."""
    if not isinstance(values, dict):
        raise TypeError("thresholds mapping must be a dict.")
    return RecommendationThresholds(
        positive_return_threshold=values.get(
            "positive_return_threshold",
            DEFAULT_POSITIVE_RETURN_THRESHOLD),
        negative_return_threshold=values.get(
            "negative_return_threshold",
            DEFAULT_NEGATIVE_RETURN_THRESHOLD),
        elevated_risk_percentile=values.get(
            "elevated_risk_percentile",
            DEFAULT_ELEVATED_RISK_PERCENTILE),
    )


# Input schema

def _require_finite(name: str, value, *, allow_none: bool = False,
                    minimum: float | None = None):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite (NaN/Inf rejected, never propagated), "
            f"got {value!r}.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}.")
    return result


@dataclass(frozen=True)
class ForecastInput:
    """One validated forecast + current risk snapshot.

    Required: ticker, as_of_date, current_price (> 0),
    predicted_log_return (finite), model_name (opaque label, kept
    exactly), annualized_volatility (finite, >= 0).
    Optional risk display metrics default to None (rendered as null,
    never invented). Sentiment defaults to unavailable per Task 3C.
    """

    ticker: str
    as_of_date: str
    current_price: float
    predicted_log_return: float
    model_name: str
    annualized_volatility: float
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    maximum_drawdown: float | None = None
    historical_var: float | None = None
    sentiment_available: bool = False
    sentiment_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("ticker must be a non-empty string.")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string.")
        try:
            parsed = pd.to_datetime(self.as_of_date)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"as_of_date is not a valid date: {self.as_of_date!r}.") from exc
        if pd.isna(parsed):
            raise ValueError(
                f"as_of_date is not a valid date: {self.as_of_date!r}.")
        object.__setattr__(self, "as_of_date", parsed.date().isoformat())
        object.__setattr__(self, "ticker", self.ticker.strip())
        object.__setattr__(
            self, "current_price",
            _require_finite("current_price", self.current_price,
                            minimum=0.0))
        if self.current_price <= 0.0:
            raise ValueError(
                f"current_price must be > 0, got {self.current_price!r}.")
        object.__setattr__(self, "predicted_log_return",
                           _require_finite("predicted_log_return",
                                           self.predicted_log_return))
        object.__setattr__(
            self, "annualized_volatility",
            _require_finite("annualized_volatility",
                            self.annualized_volatility, minimum=0.0))
        for name in ("sharpe_ratio", "sortino_ratio", "maximum_drawdown",
                     "historical_var"):
            object.__setattr__(self, name,
                               _require_finite(name, getattr(self, name),
                                               allow_none=True))
        if not isinstance(self.sentiment_available, bool):
            raise TypeError("sentiment_available must be a bool.")
        object.__setattr__(
            self, "sentiment_score",
            _require_finite("sentiment_score", self.sentiment_score,
                            allow_none=True))
        if self.sentiment_available and self.sentiment_score is None:
            raise ValueError(
                "sentiment_score is required when sentiment_available "
                "is True.")
        if not self.sentiment_available:
            # Unavailable means unavailable: drop any stray score rather
            # than recording a fabricated neutral value.
            object.__setattr__(self, "sentiment_score", None)


# Pure calculation functions

def log_to_simple_return(predicted_log_return: float) -> float:
    """Convert log return r_hat to simple return: exp(r_hat) - 1."""
    r_hat = _require_finite("predicted_log_return", predicted_log_return)
    return math.expm1(r_hat)


def predict_price(current_price: float,
                  predicted_log_return: float) -> float:
    """Predicted next price: current_price * exp(r_hat)."""
    price = _require_finite("current_price", current_price, minimum=0.0)
    if price <= 0.0:
        raise ValueError(f"current_price must be > 0, got {price!r}.")
    r_hat = _require_finite("predicted_log_return", predicted_log_return)
    return price * math.exp(r_hat)


def classify_forecast(predicted_simple_return: float,
                      thresholds: RecommendationThresholds
                      | None = None) -> str:
    """Classify a simple-return forecast (inclusive boundaries).

    strong_positive iff simple >= positive threshold; negative iff
    simple <= negative threshold; otherwise neutral.
    """
    thresholds = thresholds or default_thresholds()
    simple = _require_finite("predicted_simple_return",
                             predicted_simple_return)
    if simple >= thresholds.positive_return_threshold:
        return FORECAST_STRONG_POSITIVE
    if simple <= thresholds.negative_return_threshold:
        return FORECAST_NEGATIVE
    return FORECAST_NEUTRAL


def volatility_percentile(annualized_volatility: float,
                          universe_volatilities) -> float:
    """Cross-sectional percentile of one volatility within a universe.

    Definition (counting, deterministic): percentile(v) = fraction of
    universe volatilities at or below v. Ties share the same value; a
    single-asset universe yields 1.0. Inputs must be finite and >= 0;
    the universe must be non-empty.
    """
    vol = _require_finite("annualized_volatility", annualized_volatility,
                          minimum=0.0)
    try:
        values = [float(v) for v in universe_volatilities]
    except TypeError as exc:
        raise TypeError(
            "universe_volatilities must be an iterable of numbers.") from exc
    if not values:
        raise ValueError("universe_volatilities must be non-empty.")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, float) \
                or not math.isfinite(v) or v < 0.0:
            raise ValueError(
                "universe volatilities must be finite and >= 0.")
    return sum(1.0 for v in values if v <= vol) / len(values)


def classify_risk(risk_percentile: float,
                  thresholds: RecommendationThresholds | None = None) -> str:
    """Elevated iff percentile >= threshold (boundary inclusive)."""
    thresholds = thresholds or default_thresholds()
    pct = _require_finite("risk_percentile", risk_percentile)
    if not 0.0 <= pct <= 1.0:
        raise ValueError(
            f"risk_percentile must be between 0 and 1, got {pct!r}.")
    return (RISK_ELEVATED
            if pct >= thresholds.elevated_risk_percentile else RISK_NORMAL)


def apply_recommendation_rule(forecast_class: str, risk_class: str) -> str:
    """Apply the exact five-branch rule (no scores, no weights)."""
    if forecast_class == FORECAST_STRONG_POSITIVE:
        return BUY if risk_class == RISK_NORMAL else HOLD
    if forecast_class == FORECAST_NEUTRAL:
        return HOLD
    if forecast_class == FORECAST_NEGATIVE:
        return SELL if risk_class == RISK_ELEVATED else HOLD
    raise ValueError(f"unknown forecast_class: {forecast_class!r}.")


def build_reason(recommendation: str, forecast_class: str,
                 risk_class: str, sentiment_available: bool = False) -> str:
    """Build the deterministic human-readable reason sentence(s)."""
    try:
        base = _REASON[(recommendation, forecast_class, risk_class)]
    except KeyError as exc:
        raise ValueError(
            "inconsistent (recommendation, forecast_class, risk_class) "
            f"combination: {exc}.") from exc
    suffix = (_SENTIMENT_SUFFIX_AVAILABLE if sentiment_available
              else _SENTIMENT_SUFFIX_UNAVAILABLE)
    return f"{base} {suffix}"


# Output schema + main functions

@dataclass(frozen=True)
class Recommendation:
    """One deterministic recommendation output (see OUTPUT_COLUMNS)."""

    ticker: str
    as_of_date: str
    current_price: float
    predicted_log_return: float
    predicted_simple_return: float
    predicted_price: float
    model_name: str
    annualized_volatility: float
    risk_percentile: float
    risk_class: str
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    maximum_drawdown: float | None = None
    historical_var: float | None = None
    sentiment_available: bool = False
    sentiment_score: float | None = None
    sentiment_status: str = SENTIMENT_UNAVAILABLE
    recommendation: str = HOLD
    reason: str = ""

    def to_dict(self) -> dict:
        """Ordered dict following OUTPUT_COLUMNS (None renders as null)."""
        values = {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "current_price": self.current_price,
            "predicted_log_return": self.predicted_log_return,
            "predicted_simple_return": self.predicted_simple_return,
            "predicted_price": self.predicted_price,
            "model_name": self.model_name,
            "annualized_volatility": self.annualized_volatility,
            "risk_percentile": self.risk_percentile,
            "risk_class": self.risk_class,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "maximum_drawdown": self.maximum_drawdown,
            "historical_var": self.historical_var,
            "sentiment_available": self.sentiment_available,
            "sentiment_score": self.sentiment_score,
            "sentiment_status": self.sentiment_status,
            "recommendation": self.recommendation,
            "reason": self.reason,
        }
        return {key: values[key] for key in OUTPUT_COLUMNS}


def recommend_one(forecast: ForecastInput,
                  universe_volatilities: dict,
                  thresholds: RecommendationThresholds | None = None
                  ) -> Recommendation:
    """Recommend for one ticker against an explicit volatility universe.

    ``universe_volatilities`` maps ticker -> current annualized
    volatility and MUST contain ``forecast.ticker``; percentile and
    risk class derive from it. The caller owns timestamp safety of all
    inputs. Fully deterministic.
    """
    thresholds = thresholds or default_thresholds()
    if not isinstance(universe_volatilities, dict) or \
            not universe_volatilities:
        raise ValueError(
            "universe_volatilities must be a non-empty ticker->volatility "
            "mapping.")
    if forecast.ticker not in universe_volatilities:
        raise ValueError(
            f"ticker {forecast.ticker!r} missing from "
            "universe_volatilities.")
    simple = log_to_simple_return(forecast.predicted_log_return)
    price = predict_price(forecast.current_price,
                          forecast.predicted_log_return)
    percentile = volatility_percentile(
        forecast.annualized_volatility,
        universe_volatilities.values())
    forecast_class = classify_forecast(simple, thresholds)
    risk_class = classify_risk(percentile, thresholds)
    recommendation = apply_recommendation_rule(forecast_class, risk_class)
    reason = build_reason(recommendation, forecast_class, risk_class,
                          forecast.sentiment_available)
    return Recommendation(
        ticker=forecast.ticker,
        as_of_date=forecast.as_of_date,
        current_price=forecast.current_price,
        predicted_log_return=forecast.predicted_log_return,
        predicted_simple_return=simple,
        predicted_price=price,
        model_name=forecast.model_name,
        annualized_volatility=forecast.annualized_volatility,
        risk_percentile=percentile,
        risk_class=risk_class,
        sharpe_ratio=forecast.sharpe_ratio,
        sortino_ratio=forecast.sortino_ratio,
        maximum_drawdown=forecast.maximum_drawdown,
        historical_var=forecast.historical_var,
        sentiment_available=forecast.sentiment_available,
        sentiment_score=forecast.sentiment_score,
        sentiment_status=(SENTIMENT_AVAILABLE
                          if forecast.sentiment_available
                          else SENTIMENT_UNAVAILABLE),
        recommendation=recommendation,
        reason=reason,
    )


def _inputs_from_dataframe(frame: pd.DataFrame) -> list[ForecastInput]:
    required = {"ticker", "as_of_date", "current_price",
                "predicted_log_return", "model_name",
                "annualized_volatility"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input DataFrame missing columns: {missing}.")
    inputs = []
    for _, row in frame.iterrows():
        kwargs = {
            name: (None if (name in frame.columns and pd.isna(row[name])
                            and name not in required) else row[name])
            for name in required | {"sharpe_ratio", "sortino_ratio",
                                    "maximum_drawdown", "historical_var",
                                    "sentiment_available",
                                    "sentiment_score"}
            if name in frame.columns
        }
        if "sentiment_available" in kwargs and \
                kwargs["sentiment_available"] is not None:
            kwargs["sentiment_available"] = bool(
                kwargs["sentiment_available"])
        inputs.append(ForecastInput(**kwargs))
    return inputs


def recommend_batch(forecasts, thresholds: RecommendationThresholds | None = None,
                    universe_volatilities: dict | None = None) -> pd.DataFrame:
    """Recommend for many tickers; the batch IS the universe by default.

    Accepts a list of ForecastInput or a DataFrame with the input-schema
    columns. Percentiles are computed cross-sectionally over
    ``universe_volatilities`` when given, else over the batch's own
    volatilities (all rows share one as_of_date universe in that case;
    mixed timestamps are the caller's responsibility, documented).
    Returns a DataFrame with OUTPUT_COLUMNS in order. Deterministic.
    """
    thresholds = thresholds or default_thresholds()
    if isinstance(forecasts, pd.DataFrame):
        items = _inputs_from_dataframe(forecasts)
    else:
        items = list(forecasts)
        if not items:
            raise ValueError("forecasts must be non-empty.")
        for item in items:
            if not isinstance(item, ForecastInput):
                raise TypeError(
                    "forecasts must be ForecastInput items or a DataFrame.")
    if universe_volatilities is None:
        universe_volatilities = {item.ticker: item.annualized_volatility
                                 for item in items}
    rows = [recommend_one(item, universe_volatilities, thresholds).to_dict()
            for item in items]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# Demo (synthetic forecasts only; never repository model outputs)

def run_demo(output_dir: str | Path = "data/processed/recommendations",
             thresholds: RecommendationThresholds | None = None) -> pd.DataFrame:
    """Write a clearly labeled SYNTHETIC demo (no real tickers/prices).

    Forecasts use model_name="synthetic_demo" and DEMO_* tickers so the
    file can never be mistaken for validated investment performance.
    """
    thresholds = thresholds or default_thresholds()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = [
        # (ticker, price, log_return, volatility): one branch per rule arm.
        ("DEMO_BUY", 200.0, 0.02, 0.15),    # strong positive + normal
        ("DEMO_HOLD_RISK", 100.0, 0.03, 0.60),  # strong positive + elevated
        ("DEMO_HOLD_NEUT", 50.0, 0.002, 0.18),  # neutral
        ("DEMO_HOLD_NEG", 80.0, -0.02, 0.16),  # negative + normal
        ("DEMO_SELL", 120.0, -0.03, 0.55),  # negative + elevated
    ]
    inputs = [ForecastInput(ticker=t, as_of_date="2026-08-28",
                            current_price=p, predicted_log_return=r,
                            model_name="synthetic_demo",
                            annualized_volatility=v)
              for t, p, r, v in demo]
    frame = recommend_batch(inputs, thresholds)
    path = out_dir / "demo_recommendations_SYNTHETIC.csv"
    frame.to_csv(path, index=False)
    return frame


def main() -> pd.DataFrame:
    """CLI entry point: ``python -m src.recommendation`` (demo only)."""
    return run_demo()


if __name__ == "__main__":
    main()
