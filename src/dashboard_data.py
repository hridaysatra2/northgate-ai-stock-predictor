"""Read-only dashboard data helpers (Task 5A).

Presentation layer ONLY. This module:
  * never trains models, never optimizes portfolios, never writes artifacts;
  * only reads precomputed artifacts from data/processed/ and data/raw/;
  * returns defensive copies so callers cannot mutate cached frames;
  * fails gracefully (returns None) for missing OPTIONAL artifacts;
  * never fabricates replacement data;
  * never treats DEMO_* synthetic rows as real recommendations.

Heavy Streamlit caching wrappers live in app.py; functions here are pure
and unit-testable without a Streamlit runtime.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.integration import (
    MissingDependencyError,
    SYNTHETIC_MODEL_NAME,
    SYNTHETIC_TICKER_PREFIX,
)

# ---------------------------------------------------------------------------
# Constants (exact terminology from project docs / config.yaml)
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM", "HD", "V", "CAT", "KO"]
BENCHMARK_TICKER = "GSPC"
BENCHMARK_LABEL = "S&P 500"
RISK_REGIME_TICKER = "VIX"
RESEARCH_START = "2016-01-01"
RESEARCH_CUTOFF = "2026-08-31"

NAV_PAGES = [
    "Overview",
    "Stock Analysis",
    "Model Performance",
    "Portfolio Optimization",
    "Portfolio Backtest",
    "Recommendations",
    "Methodology & Limitations",
]

DISCLAIMER = (
    "Northgate is a research and decision-support system. Historical model "
    "and portfolio results do not guarantee future performance. "
    "Recommendations are not financial advice."
)

LIVE_UNAVAILABLE_MSG = (
    "Live forecast, live price/risk feeds, and historical sentiment feeds "
    "are not currently connected."
)

LIVE_RECOMMENDATIONS_UNAVAILABLE = "Live recommendations unavailable"

LIVE_RECOMMENDATIONS_EXPLANATION = (
    "Northgate currently lacks a connected live forecast provider and "
    "point-in-time live risk feed. The recommendation engine is implemented "
    "and tested, but production recommendations are not currently generated."
)

SYNTHETIC_DEMO_LABEL = "SYNTHETIC DEMO — NOT A REAL RECOMMENDATION"

HISTORICAL_BACKTEST_LABEL = "Historical rolling backtest"

BACKTEST_ASSUMPTIONS = (
    "252-trading-day estimation window; monthly rebalancing; "
    "weights estimated before each rebalance; no transaction costs; "
    "historical simulation — not a forecast."
)

# Synthetic-demo markers: single source of truth in src.integration
# (SYNTHETIC_TICKER_PREFIX = "DEMO_", SYNTHETIC_MODEL_NAME = "synthetic_demo").

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_PATHS = {
    # EDA (Task 2A; outputs may be absent — see load_adf_results)
    "adf_results": "data/processed/eda/tables/adf_results.csv",
    # features / raw
    "feature_metadata": "data/processed/features/feature_metadata.csv",
    "dataset_summary": "data/processed/deep_learning/dataset_summary.csv",
    # classical + evaluation (Task 2D)
    "model_metrics": "data/processed/models/model_metrics.csv",
    "evaluation_metrics": "data/processed/evaluation/evaluation_metrics.csv",
    "aggregate_summary": "data/processed/evaluation/aggregate_model_summary.csv",
    "best_by_ticker": "data/processed/evaluation/best_model_by_ticker.csv",
    "baseline_comparison": "data/processed/evaluation/baseline_comparison.csv",
    "evaluation_summary_txt": "data/processed/evaluation/evaluation_summary.txt",
    # deep learning (Task 3B)
    "dl_metrics": "data/processed/deep_learning_training/dl_metrics.csv",
    "dl_best_models": "data/processed/deep_learning_training/dl_best_models.csv",
    "dl_config": "data/processed/deep_learning_training/config.json",
    # portfolio (Tasks 4C / 4D)
    "key_portfolios": "data/processed/portfolio_frontier/key_portfolios.csv",
    "efficient_frontier": "data/processed/portfolio_frontier/efficient_frontier.csv",
    "monte_carlo": "data/processed/portfolio_frontier/monte_carlo_portfolios.csv",
    "frontier_metadata": "data/processed/portfolio_frontier/frontier_metadata.json",
    "strategy_metrics": "data/processed/portfolio_backtest/strategy_metrics.csv",
    "daily_values": "data/processed/portfolio_backtest/daily_portfolio_values.csv",
    "daily_returns": "data/processed/portfolio_backtest/daily_portfolio_returns.csv",
    "rebalance_weights": "data/processed/portfolio_backtest/rebalance_weights.csv",
    "backtest_summary": "data/processed/portfolio_backtest/backtest_summary.json",
    "backtest_methodology": "data/processed/portfolio_backtest/BACKTEST_METHODOLOGY.md",
    # recommendations (Task 4E, synthetic demo only)
    "demo_recommendations": "data/processed/recommendations/demo_recommendations_SYNTHETIC.csv",
    "recommendation_methodology": "data/processed/recommendations/RECOMMENDATION_METHODOLOGY.md",
    # integration / sentiment
    "integration_audit": "data/processed/integration/INTEGRATION_AUDIT.md",
    "news_availability": "data/processed/news_availability/NEWS_DATA_AVAILABILITY.md",
}


# ---------------------------------------------------------------------------
# Path / status helpers
# ---------------------------------------------------------------------------

def get_artifact_paths() -> dict:
    """Return a copy of the known artifact-path map (relative to repo root)."""
    return dict(ARTIFACT_PATHS)


def resolve_path(relpath: str) -> str:
    """Resolve a repo-relative path to an absolute path."""
    return os.path.join(REPO_ROOT, relpath)


def artifact_exists(key: str) -> bool:
    """True iff the artifact for key exists on disk."""
    rel = ARTIFACT_PATHS.get(key)
    if rel is None:
        return False
    return os.path.isfile(resolve_path(rel))


def artifact_status() -> dict:
    """Map artifact key -> 'Available' | 'Unavailable' (never fabricates)."""
    return {
        key: ("Available" if artifact_exists(key) else "Unavailable")
        for key in ARTIFACT_PATHS
    }


# ---------------------------------------------------------------------------
# Safe read-only loaders (return copies; None when optional artifact missing)
# ---------------------------------------------------------------------------

def load_csv_artifact(key: str):
    """Load a CSV artifact by key; return a fresh DataFrame copy or None.

    Returns None (instead of raising) when the file is absent so the UI can
    show an 'Unavailable' state. Never synthesizes replacement rows.
    """
    rel = ARTIFACT_PATHS.get(key)
    if rel is None:
        return None
    path = resolve_path(rel)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df.copy(deep=True)


def load_json_artifact(key: str):
    """Load a JSON artifact by key; return dict or None if missing."""
    rel = ARTIFACT_PATHS.get(key)
    if rel is None:
        return None
    path = resolve_path(rel)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_text_artifact(key: str):
    """Load a text/markdown artifact by key; return str or None if missing."""
    rel = ARTIFACT_PATHS.get(key)
    if rel is None:
        return None
    path = resolve_path(rel)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_raw_prices(ticker: str):
    """Load raw market parquet for ticker; None if missing. Read-only copy.

    Rows are clipped to RESEARCH_CUTOFF so research views can never extend
    past the frozen cutoff even if a raw file later contains newer rows.
    """
    path = resolve_path(f"data/raw/market/{ticker}.parquet")
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path)
    out = df.copy(deep=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)
    return clip_to_cutoff(out)


def load_feature_frame(ticker: str):
    """Load processed feature parquet for ticker; None if missing.

    Clipped to RESEARCH_CUTOFF (see load_raw_prices).
    """
    path = resolve_path(f"data/processed/features/{ticker}.parquet")
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path)
    out = df.copy(deep=True)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)
    return clip_to_cutoff(out)


def clip_to_cutoff(df: pd.DataFrame, date_col: str = "date"):
    """Return a copy of df restricted to date_col <= RESEARCH_CUTOFF.

    Display-guard only: raw files are never modified. Frames without a
    date column are returned as an unchanged copy.
    """
    if df is None:
        return None
    out = df.copy(deep=True)
    if date_col not in out.columns:
        return out
    cutoff = pd.to_datetime(RESEARCH_CUTOFF)
    return out.loc[pd.to_datetime(out[date_col]) <= cutoff].reset_index(drop=True)


def format_directional_accuracy(value: float) -> str:
    """Format a directional-accuracy proportion as a percentage string.

    Directional accuracy is a rate in [0, 1]: 0.509 -> '50.9%'.
    Display-only conversion; never write back to artifacts.
    """
    return f"{100.0 * float(value):.1f}%"


ADF_UNAVAILABLE_REASON = (
    "ADF results unavailable: no Task 2A EDA output found at "
    "'data/processed/eda/tables/adf_results.csv' (the data/processed/eda/ "
    "directory is not present in this checkout). No statistic is invented; "
    "ADF can be reproduced with the frozen helper src.eda.adf_summary "
    "(statistic, p-value, lags). Interpretation if shown: high p-value "
    "fails to reject the unit-root null; low p-value is evidence against "
    "the unit-root null."
)


def load_adf_results():
    """Load Task 2A ADF results if a compatible artifact exists.

    Returns (DataFrame or None, reason str). Never recomputes ADF and
    never fabricates rows: when the artifact is absent, returns
    (None, ADF_UNAVAILABLE_REASON).
    """
    rel = ARTIFACT_PATHS.get("adf_results")
    path = resolve_path(rel) if rel else None
    if path and os.path.isfile(path):
        df = pd.read_csv(path)
        return df.copy(deep=True), "Available"
    return None, ADF_UNAVAILABLE_REASON


# ---------------------------------------------------------------------------
# Synthetic-data safeguards
# ---------------------------------------------------------------------------

def is_synthetic_ticker(ticker) -> bool:
    """True iff ticker carries the DEMO_ synthetic prefix."""
    return isinstance(ticker, str) and ticker.startswith(SYNTHETIC_TICKER_PREFIX)


def is_synthetic_record(record: dict) -> bool:
    """True iff a recommendation record is explicitly labeled synthetic demo."""
    if not isinstance(record, dict):
        return False
    ticker = str(record.get("ticker", ""))
    model = str(record.get("model_name", ""))
    return ticker.startswith(SYNTHETIC_TICKER_PREFIX) or model == SYNTHETIC_MODEL_NAME


def filter_real_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every DEMO_* / synthetic_demo row; never empty-fills.

    Returns a fresh copy. An empty result simply means no real
    recommendations exist (the production state), not an error.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy(deep=True)
    if "ticker" in out.columns:
        mask = out["ticker"].astype(str).str.startswith(SYNTHETIC_TICKER_PREFIX)
        out = out.loc[~mask].copy(deep=True)
    if "model_name" in out.columns:
        mask = out["model_name"].astype(str) == SYNTHETIC_MODEL_NAME
        out = out.loc[~mask].copy(deep=True)
    return out.reset_index(drop=True)


def label_demo_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the prominent synthetic-demo label column (presentation only)."""
    if df is None:
        return pd.DataFrame()
    out = df.copy(deep=True)
    out["demo_label"] = SYNTHETIC_DEMO_LABEL
    return out


def get_live_recommendation_status() -> dict:
    """Canonical live-recommendation state: unavailable, with explanation."""
    return {
        "status": LIVE_RECOMMENDATIONS_UNAVAILABLE,
        "explanation": LIVE_RECOMMENDATIONS_EXPLANATION,
        "live_forecast_connected": False,
        "live_price_feed_connected": False,
        "live_risk_feed_connected": False,
        "sentiment_available": False,
    }


def try_generate_live_recommendation(forecast=None, risk=None, **kwargs):
    """Never generate a recommendation when live inputs are absent.

    The dashboard has no live forecast provider and no point-in-time live
    risk feed (Task 4F). This helper therefore always raises a
    MissingDependencyError-style ValueError instead of fabricating output.
    """
    if forecast is None:
        raise MissingDependencyError(
            "no live forecast provider connected: production recommendations "
            "are not currently generated (Task 4F)."
        )
    if risk is None:
        raise MissingDependencyError(
            "no point-in-time live risk feed connected: production "
            "recommendations are not currently generated (Task 4F)."
        )
    raise MissingDependencyError(
        "live recommendations unavailable: caller-supplied forecast/risk "
        "are not connected to a validated live provider."
    )


# ---------------------------------------------------------------------------
# Presentation-only derived values (documented conversions, no methodology change)
# ---------------------------------------------------------------------------

def compute_drawdown(value_series: pd.Series) -> pd.Series:
    """Drawdown fraction (<= 0) of a value curve. Display-only derivation."""
    s = pd.to_numeric(value_series, errors="coerce").astype(float)
    running_max = s.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = s / running_max - 1.0
    dd[running_max == 0] = 0.0
    return dd.fillna(0.0)


def normalized_price(series: pd.Series, base: float | None = None) -> pd.Series:
    """Rebase a price series to 100 at first observation (display only)."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    b = float(s.iloc[0]) if base is None else float(base)
    if b == 0 or not np.isfinite(b):
        return pd.Series(np.nan, index=s.index)
    return s / b * 100.0


def summarize_stock_history(raw_df: pd.DataFrame) -> dict:
    """Historical research statistics for the Stock Analysis page.

    Expects a raw market frame with adj_close + date. Returns plain
    scalars; empty dict when input is missing/empty. Labeled historical.
    """
    if raw_df is None or len(raw_df) == 0 or "adj_close" not in raw_df.columns:
        return {}
    df = raw_df.copy(deep=True)
    prices = pd.to_numeric(df["adj_close"], errors="coerce").dropna()
    if len(prices) < 2:
        return {}
    log_ret = np.log(prices / prices.shift(1)).dropna()
    out = {
        "n_observations": int(len(df)),
        "first_date": str(pd.to_datetime(df["date"]).min().date()) if "date" in df.columns else None,
        "last_date": str(pd.to_datetime(df["date"]).max().date()) if "date" in df.columns else None,
        "last_adj_close": float(prices.iloc[-1]),
        "mean_daily_log_return": float(log_ret.mean()),
        "std_daily_log_return": float(log_ret.std(ddof=1)),
        "skewness": float(log_ret.skew()),
        "kurtosis": float(log_ret.kurt()),
        "min_daily_log_return": float(log_ret.min()),
        "max_daily_log_return": float(log_ret.max()),
    }
    return out


def get_research_cutoff() -> str:
    """Return the frozen research cutoff date string."""
    return RESEARCH_CUTOFF
