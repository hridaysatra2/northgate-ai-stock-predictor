"""Leakage-free feature engineering for Northgate AI Stock Predictor.

Reads raw Parquet files under ``data/raw/market/`` (NEVER writes there)
and builds supervised feature datasets under ``data/processed/features/``.

Prediction-time rule enforced everywhere:
    Every feature on row/date t uses ONLY information available through
    date t (trailing windows, lagged values, calendar fields derived from
    t alone). The target on row t is the 1-trading-day-ahead log return
    log(adj_{t+1} / adj_t), built with a backward shift so the future price
    never leaks into the feature row.

Leakage controls:
  - All rolling statistics use trailing windows with
    ``min_periods=window`` (value at t covers rows t-window+1..t only).
  - Lags/momentum use ``.shift(k)`` (rows t-k..t only).
  - Market series (GSPC/VIX) are pre-computed on their own histories and
    left-merged onto equity dates; no forward-fill, no synthetic dates.
  - ``month_end`` is the calendar ``is_month_end`` flag, a pure function
    of t (NOT the observed last trading day of the month, which would peek
    at later rows within the month).
  - Research cutoff ``date <= 2026-08-31`` is enforced from config; the
    target's t+1 requirement additionally drops the last raw row.
  - No interpolation, no filling of prices/returns/volumes. Warm-up rows
    with unavailable features/target are dropped from the final files.
  - No scaling here (handled later, after chronological splits).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
SQRT_252 = float(np.sqrt(TRADING_DAYS_PER_YEAR))

RAW_DIR = Path("data/raw/market")
FEATURES_DIR = Path("data/processed/features")
METADATA_PATH = FEATURES_DIR / "feature_metadata.csv"

# Maximum trailing lookback in trading days (vol_60d needs 60 log returns,
# each needing one prior price, i.e. adj[i-60..i] for row i).
MAX_LOOKBACK = 60

EQUITY_FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_lag_2",
    "log_return_lag_5",
    "log_return_lag_10",
    "sma_5",
    "sma_20",
    "sma_50",
    "price_to_sma20",
    "price_to_sma50",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "vol_5d",
    "vol_5d_ann",
    "vol_20d",
    "vol_20d_ann",
    "vol_60d",
    "vol_60d_ann",
    "volume_change",
    "volume_sma_20",
    "volume_ratio_20",
    "day_of_week",
    "month",
    "month_end",
    "sp500_log_return_1d",
    "sp500_momentum_5d",
    "sp500_vol_20d",
    "vix_level",
    "vix_change_1d",
    "vix_change_5d",
]

TARGET_COLUMN = "target_log_return_1d"


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division yielding NaN (never inf) on zero/NaN denom."""
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den.mask(den == 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _log_return(adj: pd.Series) -> pd.Series:
    """log(adj_t / adj_{t-1}); NaN first row; NaN on non-positive prices."""
    adj = pd.to_numeric(adj, errors="coerce").astype(float).mask(
        pd.to_numeric(adj, errors="coerce").astype(float) <= 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(adj / adj.shift(1))
    return out.replace([np.inf, -np.inf], np.nan)


def _trailing_sma(s: pd.Series, window: int) -> pd.Series:
    """Trailing simple moving average (rows t-window+1..t only)."""
    return pd.to_numeric(s, errors="coerce").astype(float).rolling(
        window=window, min_periods=window
    ).mean()


def _trailing_std(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling std (rows t-window+1..t only; nothing before)."""
    return pd.to_numeric(s, errors="coerce").astype(float).rolling(
        window=window, min_periods=window
    ).std()


# ---------------------------------------------------------------------------
# Feature builders (pure; no I/O)
# ---------------------------------------------------------------------------

def build_own_features(dates: pd.Series, adj: pd.Series,
                       volume: pd.Series) -> pd.DataFrame:
    """Equity-local features for one series, all available at time t.

    Lag-name note: ``log_return_lag_2`` is the 1-day log return shifted by
    1 row (value at t equals the return of t-1); ``lag_5``/``lag_10`` are
    shifted by 4/9 rows respectively. Names are historical window labels,
    not shift amounts; values are unchanged.
    """
    adj = pd.to_numeric(adj, errors="coerce").astype(float)
    volume = pd.to_numeric(volume, errors="coerce").astype(float)
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)

    logret = _log_return(adj)
    out = pd.DataFrame({"date": dates.reset_index(drop=True)})
    out["log_return_1d"] = logret.values
    # lag_1 of the 1-day log return IS log_return_1d itself, so it is not
    # duplicated; deeper lags are explicit shifts (rows <= t only).
    out["log_return_lag_2"] = logret.shift(1).values
    out["log_return_lag_5"] = logret.shift(4).values
    out["log_return_lag_10"] = logret.shift(9).values

    out["sma_5"] = _trailing_sma(adj, 5).values
    out["sma_20"] = _trailing_sma(adj, 20).values
    out["sma_50"] = _trailing_sma(adj, 50).values
    out["price_to_sma20"] = (_safe_divide(adj, out["sma_20"]) - 1.0).values
    out["price_to_sma50"] = (_safe_divide(adj, out["sma_50"]) - 1.0).values

    out["momentum_5d"] = (_safe_divide(adj, adj.shift(5)) - 1.0).values
    out["momentum_10d"] = (_safe_divide(adj, adj.shift(10)) - 1.0).values
    out["momentum_20d"] = (_safe_divide(adj, adj.shift(20)) - 1.0).values

    for w in (5, 20, 60):
        std = _trailing_std(logret, w)
        out[f"vol_{w}d"] = std.values
        out[f"vol_{w}d_ann"] = (std * SQRT_252).values

    out["volume_change"] = (
        _safe_divide(volume, volume.shift(1)) - 1.0
    ).values
    out["volume_sma_20"] = _trailing_sma(volume, 20).values
    out["volume_ratio_20"] = _safe_divide(
        volume, out["volume_sma_20"]
    ).values

    # Pure functions of t alone (no other rows involved).
    out["day_of_week"] = dates.dt.dayofweek.values
    out["month"] = dates.dt.month.values
    out["month_end"] = dates.dt.is_month_end.astype(int).values
    return out


def build_market_features(sp500: pd.DataFrame,
                           vix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market-context features computed on market histories, then alignable.

    Each value at date d uses only that market series' rows <= d.
    """
    sp = sp500.sort_values("date").reset_index(drop=True)
    vx = vix.sort_values("date").reset_index(drop=True)

    sp_logret = _log_return(sp["adj_close"])
    sp_mkt = pd.DataFrame({"date": pd.to_datetime(sp["date"])})
    sp_mkt["sp500_log_return_1d"] = sp_logret.values
    sp_mkt["sp500_momentum_5d"] = (
        _safe_divide(sp["adj_close"], sp["adj_close"].shift(5)) - 1.0
    ).values
    sp_mkt["sp500_vol_20d"] = _trailing_std(sp_logret, 20).values

    vx_close = pd.to_numeric(vx["close"], errors="coerce").astype(float)
    vx_mkt = pd.DataFrame({"date": pd.to_datetime(vx["date"])})
    vx_mkt["vix_level"] = vx_close.values
    vx_mkt["vix_change_1d"] = (
        _safe_divide(vx_close, vx_close.shift(1)) - 1.0
    ).values
    vx_mkt["vix_change_5d"] = (
        _safe_divide(vx_close, vx_close.shift(5)) - 1.0
    ).values
    return sp_mkt, vx_mkt


def build_target(adj: pd.Series) -> pd.Series:
    """1-trading-day-ahead log return for row t: log(adj_{t+1} / adj_t)."""
    adj = pd.to_numeric(adj, errors="coerce").astype(float).mask(
        pd.to_numeric(adj, errors="coerce").astype(float) <= 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(adj.shift(-1) / adj)
    out.name = TARGET_COLUMN
    return out.replace([np.inf, -np.inf], np.nan)


def build_supervised_frame(
    ticker: str,
    raw: pd.DataFrame,
    sp_market: pd.DataFrame,
    vix_market: pd.DataFrame,
    cutoff: str,
    include_target: bool = True,
    include_volume_features: bool = True,
) -> pd.DataFrame:
    """Assemble the final leakage-free frame for one ticker.

    Steps: enforce cutoff, sort, build own features, left-merge market
    features by date (no fill), attach shifted target, drop any row with
    unavailable features/target.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.to_datetime(cutoff)]
    df = df.sort_values("date").reset_index(drop=True)

    feats = build_own_features(df["date"], df["adj_close"], df["volume"])
    feats = feats.merge(sp_market, on="date", how="left")
    feats = feats.merge(vix_market, on="date", how="left")
    feats.insert(1, "ticker", ticker)

    required = list(EQUITY_FEATURE_COLUMNS)
    if not include_volume_features:
        # VIX has no traded volume (raw volume is 0); volume features are
        # undefined and therefore excluded rather than fabricated.
        required = [c for c in required if not c.startswith("volume")]
    if include_target:
        feats[TARGET_COLUMN] = build_target(df["adj_close"]).values
        required = required + [TARGET_COLUMN]

    cols = ["date", "ticker"] + required
    feats = feats[cols]
    # Drop warm-up / target-unavailable rows; never fill.
    feats = feats.dropna(subset=required).reset_index(drop=True)
    return feats


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def feature_metadata_rows() -> list[dict]:
    """One metadata row per feature/target documenting leakage safety."""
    trailing = ("trailing window rows t-w+1..t; min_periods=w; no future use")
    rows = [
        ("log_return_1d", "return", "log(adj_t/adj_{t-1})", 1, False,
         "adj_close", "uses rows t-1..t only"),
        ("log_return_lag_2", "return", "log_return_1d shifted by 1", 2, False,
         "adj_close", "value at t equals return of t-1"),
        ("log_return_lag_5", "return", "log_return_1d shifted by 4", 5, False,
         "adj_close", "value at t equals return of t-4"),
        ("log_return_lag_10", "return", "log_return_1d shifted by 9", 10,
         False, "adj_close", "value at t equals return of t-9"),
        ("sma_5", "trend", "mean(adj t-4..t)", 5, False, "adj_close",
         trailing),
        ("sma_20", "trend", "mean(adj t-19..t)", 20, False, "adj_close",
         trailing),
        ("sma_50", "trend", "mean(adj t-49..t)", 50, False, "adj_close",
         trailing),
        ("price_to_sma20", "trend", "adj_t/sma_20(t)-1", 20, False,
         "adj_close", "both inputs known at t"),
        ("price_to_sma50", "trend", "adj_t/sma_50(t)-1", 50, False,
         "adj_close", "both inputs known at t"),
        ("momentum_5d", "momentum", "adj_t/adj_{t-5}-1", 5, False,
         "adj_close", "uses rows t-5..t only"),
        ("momentum_10d", "momentum", "adj_t/adj_{t-10}-1", 10, False,
         "adj_close", "uses rows t-10..t only"),
        ("momentum_20d", "momentum", "adj_t/adj_{t-20}-1", 20, False,
         "adj_close", "uses rows t-20..t only"),
        ("vol_5d", "volatility", "std(logret t-4..t)", 5, False,
         "adj_close", trailing + " over past log returns"),
        ("vol_5d_ann", "volatility", "vol_5d*sqrt(252)", 5, True,
         "adj_close", "annualized trailing measure"),
        ("vol_20d", "volatility", "std(logret t-19..t)", 20, False,
         "adj_close", trailing + " over past log returns"),
        ("vol_20d_ann", "volatility", "vol_20d*sqrt(252)", 20, True,
         "adj_close", "annualized trailing measure"),
        ("vol_60d", "volatility", "std(logret t-59..t)", 60, False,
         "adj_close", trailing + " over past log returns"),
        ("vol_60d_ann", "volatility", "vol_60d*sqrt(252)", 60, True,
         "adj_close", "annualized trailing measure"),
        ("volume_change", "volume", "vol_t/vol_{t-1}-1", 1, False,
         "volume", "uses rows t-1..t only; NaN never filled"),
        ("volume_sma_20", "volume", "mean(vol t-19..t)", 20, False,
         "volume", trailing),
        ("volume_ratio_20", "volume", "vol_t/volume_sma_20(t)", 20, False,
         "volume", "both inputs known at t"),
        ("day_of_week", "calendar", "Monday=0..Friday=4 from date t", 0,
         False, "date", "pure function of t"),
        ("month", "calendar", "1..12 from date t", 0, False, "date",
         "pure function of t"),
        ("month_end", "calendar", "calendar is_month_end(date t)", 0, False,
         "date", "pure function of t; NOT observed last trading day"),
        ("sp500_log_return_1d", "market", "log(sp_adj_t/sp_adj_{t-1})", 1,
         False, "GSPC adj_close",
         "computed on GSPC history; left-merged by date; no fill"),
        ("sp500_momentum_5d", "market", "sp_adj_t/sp_adj_{t-5}-1", 5, False,
         "GSPC adj_close",
         "computed on GSPC history; left-merged by date; no fill"),
        ("sp500_vol_20d", "market", "std(sp logret t-19..t)", 20, False,
         "GSPC adj_close",
         "computed on GSPC history; left-merged by date; no fill"),
        ("vix_level", "market", "VIX close_t", 0, False, "VIX close",
         "same-date market observation only"),
        ("vix_change_1d", "market", "vix_t/vix_{t-1}-1", 1, False,
         "VIX close", "computed on VIX history; merged by date; no fill"),
        ("vix_change_5d", "market", "vix_t/vix_{t-5}-1", 5, False,
         "VIX close", "computed on VIX history; merged by date; no fill"),
        (TARGET_COLUMN, "target", "log(adj_{t+1}/adj_t) via shift(-1)",
         1, False, "adj_close",
         "future price by construction; belongs to row t; last row dropped"),
    ]
    return [
        {
            "feature_name": name,
            "category": cat,
            "formula": formula,
            "lookback_window": window,
            "annualized": ann,
            "source_columns": src,
            "maximum_lookback": MAX_LOOKBACK,
            "leakage_notes": notes,
        }
        for name, cat, formula, window, ann, src, notes in rows
    ]


# ---------------------------------------------------------------------------
# I/O + CLI
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load universe and research dates from config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_filename(ticker: str) -> str:
    """Strip '^' for safe filenames (^GSPC -> GSPC)."""
    return ticker.replace("^", "")


def read_raw_ticker(ticker: str, raw_dir: str | Path) -> pd.DataFrame:
    """Read-only load of one raw Parquet file, sorted by date."""
    df = pd.read_parquet(Path(raw_dir) / (sanitize_filename(ticker)
                                          + ".parquet"))
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def run_features(
    config_path: str | Path = "config.yaml",
    raw_dir: str | Path = RAW_DIR,
    output_dir: str | Path = FEATURES_DIR,
) -> dict:
    """Generate supervised feature datasets + metadata (deterministic).

    Reads raw files only; writes one Parquet per ticker plus
    feature_metadata.csv. No downloads, no randomness.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    cfg = load_config(config_path)
    universe = cfg["universe"]
    equities: list[str] = list(universe["tickers"])
    benchmark: str = universe["benchmark"]
    risk_ticker: str = universe["risk_regime_ticker"]
    cutoff: str = universe["research_cutoff_date"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_raw = read_raw_ticker(benchmark, raw_dir)
    vix_raw = read_raw_ticker(risk_ticker, raw_dir)
    sp_market, vix_market = build_market_features(sp_raw, vix_raw)

    row_counts: dict[str, int] = {}
    # Supervised files for equities + benchmark (target included).
    for ticker in equities + [benchmark]:
        raw = sp_raw if ticker == benchmark else read_raw_ticker(
            ticker, raw_dir)
        frame = build_supervised_frame(ticker, raw, sp_market, vix_market,
                                       cutoff, include_target=True)
        frame.to_parquet(
            out_dir / (sanitize_filename(ticker) + ".parquet"), index=False)
        row_counts[ticker] = len(frame)
        logger.info("%s: %d supervised rows (%s..%s)", ticker, len(frame),
                    frame["date"].min().date(), frame["date"].max().date())

    # VIX diagnostics file: features only, NO supervised target.
    vix_frame = build_supervised_frame(
        risk_ticker, vix_raw, sp_market, vix_market, cutoff,
        include_target=False, include_volume_features=False)
    vix_frame.to_parquet(
        out_dir / (sanitize_filename(risk_ticker) + ".parquet"), index=False)
    row_counts[risk_ticker] = len(vix_frame)

    pd.DataFrame(feature_metadata_rows()).to_csv(
        out_dir / METADATA_PATH.name, index=False)

    print("=" * 72)
    print("FEATURES COMPLETE (leakage-free; no modelling)")
    print(f"Datasets -> {out_dir} | metadata -> {out_dir / METADATA_PATH.name}")
    for t, n in row_counts.items():
        print(f"  {t}: {n} rows")
    print("=" * 72)
    return {"row_counts": row_counts, "output_dir": str(out_dir)}


def main(config_path: str | Path = "config.yaml") -> dict:
    """CLI entry point: ``python -m src.features``."""
    return run_features(config_path=config_path)


if __name__ == "__main__":
    main()
