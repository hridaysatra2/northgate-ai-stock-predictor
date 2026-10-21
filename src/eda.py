"""Exploratory Data Analysis (EDA) for Northgate AI Stock Predictor.

Reads the existing raw Parquet files under ``data/raw/market/`` (NEVER
writes there) and produces descriptive statistics, tables and plots under
``data/processed/eda/{tables,plots}/``.

Scope is strictly descriptive EDA: no model features, no targets, no
train/test splits, no sentiment, no portfolio logic, no dashboard.

Leakage rules respected here:
  - All rolling calculations use trailing windows only (``min_periods``
    equals the window length), so a value at time t never uses rows after t.
  - No forward/backward filling and no interpolation of financial series.
  - Correlation uses aligned (inner-joined) dates only, pairwise complete.
  - The 1-day-ahead log-return target is NOT created in this module.

Notes on measures:
  - Log returns use Adjusted Close: log(adj_t / adj_{t-1}).
  - Annualized volatility = rolling std of daily log returns * sqrt(252).
  - VIX (implied volatility, in percent) and realized rolling volatility
    are DIFFERENT measures; they are plotted together only for visual
    market context, with realized vol scaled to percent (x100).
  - Normalized prices (first adj_close -> 100) are descriptive
    normalization for visual comparison, not a prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; no interactive display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
VOL_WINDOW = 20
SMA_WINDOWS = (20, 50, 200)

RAW_DIR = Path("data/raw/market")
EDA_DIR = Path("data/processed/eda")
TABLES_DIR = EDA_DIR / "tables"
PLOTS_DIR = EDA_DIR / "plots"


# ---------------------------------------------------------------------------
# Pure deterministic helpers (unit-tested, no I/O, no network)
# ---------------------------------------------------------------------------

def compute_log_returns(adj_close: pd.Series) -> pd.Series:
    """Daily log returns: log(adj_t / adj_{t-1}) using Adjusted Close.

    First observation is NaN. Non-positive prices yield NaN (no inf).
    """
    adj = pd.to_numeric(adj_close, errors="coerce").astype(float)
    adj = adj.mask(adj <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(adj / adj.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns.name = "log_return"
    return returns


def compute_simple_returns(adj_close: pd.Series) -> pd.Series:
    """Daily simple returns: adj_t / adj_{t-1} - 1 (for wealth/drawdown)."""
    adj = pd.to_numeric(adj_close, errors="coerce").astype(float)
    out = adj / adj.shift(1) - 1.0
    out.name = "simple_return"
    return out


def normalize_to_100(series: pd.Series) -> pd.Series:
    """Descriptive normalization: first value -> 100 (visual comparison)."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    first = s.iloc[0]
    out = s / first * 100.0
    out.name = "normalized_100"
    return out


def rolling_realized_vol(
    log_returns: pd.Series, window: int = VOL_WINDOW
) -> tuple[pd.Series, pd.Series]:
    """Trailing rolling std of log returns and annualized version (*sqrt(252)).

    ``min_periods=window`` so no value exists before enough observations.
    Trailing-only window: value at t uses rows t-window+1..t (never future).
    """
    r = pd.to_numeric(log_returns, errors="coerce").astype(float)
    roll_std = r.rolling(window=window, min_periods=window).std()
    roll_std.name = "rolling_std"
    ann = roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    ann.name = "rolling_vol_annualized"
    return roll_std, ann


def maximum_drawdown(simple_returns: pd.Series) -> float:
    """Maximum drawdown from simple returns via wealth = cumprod(1+r)."""
    r = pd.to_numeric(simple_returns, errors="coerce").astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of aligned daily log returns (pairwise complete).

    Rows are dates (inner-joined beforehand by the caller); no filling.
    """
    return returns_wide.corr(method="pearson")


def align_series_by_date(
    values_by_ticker: dict[str, pd.Series],
    dates_by_ticker: dict[str, pd.Series],
) -> pd.DataFrame:
    """Inner-join per-ticker series on actual dates (no positional pairing).

    Each series is indexed by its own dates, then concatenated with
    ``join="inner"`` so only genuinely matched dates survive. No filling.
    """
    indexed = {}
    for ticker, values in values_by_ticker.items():
        dates = pd.to_datetime(dates_by_ticker[ticker])
        vals = pd.Series(
            np.asarray(values), index=pd.DatetimeIndex(dates), name=ticker
        )
        if vals.index.duplicated().any():
            raise ValueError(
                f"duplicate dates for ticker {ticker!r}; cannot align."
            )
        indexed[ticker] = vals
    wide = pd.concat(indexed, axis=1, join="inner").sort_index()
    wide.index.name = "date"
    return wide


def data_quality_summary(ticker: str, df: pd.DataFrame) -> dict:
    """Descriptive data-quality facts for one raw ticker frame (read-only)."""
    dates = pd.to_datetime(df["date"])
    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    summary: dict = {
        "ticker": ticker,
        "n_observations": int(len(df)),
        "first_date": dates.min().date().isoformat() if len(df) else None,
        "last_date": dates.max().date().isoformat() if len(df) else None,
        "n_duplicate_dates": int(dates.duplicated().sum()),
        "is_monotonic_increasing": bool(dates.is_monotonic_increasing),
    }
    for c in df.columns:
        summary[f"missing_{c}"] = missing[c]
    for c in numeric:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            summary[f"min_{c}"] = float(s.min())
            summary[f"max_{c}"] = float(s.max())
    return summary


def return_statistics(ticker: str, log_returns: pd.Series) -> dict:
    """Descriptive stats + tail percentiles of a log-return series."""
    r = pd.to_numeric(log_returns, errors="coerce").astype(float).dropna()
    return {
        "ticker": ticker,
        "count": int(r.count()),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "p01": float(r.quantile(0.01)),
        "p05": float(r.quantile(0.05)),
        "q25": float(r.quantile(0.25)),
        "median": float(r.median()),
        "q75": float(r.quantile(0.75)),
        "p95": float(r.quantile(0.95)),
        "p99": float(r.quantile(0.99)),
        "max": float(r.max()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),  # Fisher (excess) kurtosis, pandas default
    }


def adf_summary(ticker: str, series_name: str, series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test result as a plain descriptive record.

    Returns the statistic, p-value, observation count and critical values.
    No stationarity verdict is imposed here.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float).dropna()
    stat, p_value, _lag, nobs, crit, _icbest = adfuller(
        s, autolag="AIC", result_object=False
    )
    return {
        "ticker": ticker,
        "series": series_name,
        "adf_statistic": float(stat),
        "p_value": float(p_value),
        "n_observations": int(nobs),
        "crit_1pct": float(crit.get("1%")),
        "crit_5pct": float(crit.get("5%")),
        "crit_10pct": float(crit.get("10%")),
    }


def calendar_return_stats(
    dates: pd.Series, log_returns: pd.Series
) -> pd.DataFrame:
    """Mean daily log return + sample size by day-of-week and month.

    Purely descriptive; no significance claims.
    """
    frame = pd.DataFrame(
        {"date": pd.to_datetime(dates), "log_return": log_returns}
    ).dropna()
    dow = frame.copy()
    dow["group"] = "dow"
    dow["group_value"] = dow["date"].dt.day_name()
    mon = frame.copy()
    mon["group"] = "month"
    mon["group_value"] = mon["date"].dt.month.astype(str).str.zfill(2)
    both = pd.concat([dow, mon], ignore_index=True)
    out = (
        both.groupby(["group", "group_value"])["log_return"]
        .agg(mean_log_return="mean", n="size")
        .reset_index()
    )
    return out


def cross_asset_row(
    ticker: str, dates: pd.Series, adj_close: pd.Series, log_returns: pd.Series
) -> dict:
    """One descriptive cross-asset summary row for an equity."""
    adj = pd.to_numeric(adj_close, errors="coerce").astype(float)
    r = pd.to_numeric(log_returns, errors="coerce").astype(float).dropna()
    simple = compute_simple_returns(adj).dropna()
    n = int(r.count())
    cumulative_return = float(adj.iloc[-1] / adj.iloc[0] - 1.0)
    ann_mean = float(r.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    return {
        "ticker": ticker,
        "observations": int(len(adj)),
        "return_observations": n,
        "first_date": pd.to_datetime(dates.iloc[0]).date().isoformat(),
        "last_date": pd.to_datetime(dates.iloc[-1]).date().isoformat(),
        "cumulative_return": cumulative_return,
        "annualized_mean_log_return": ann_mean,
        "annualized_volatility": ann_vol,
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "maximum_drawdown": maximum_drawdown(simple),
    }


# ---------------------------------------------------------------------------
# I/O helpers (read raw only; write processed only)
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load tickers and research dates from config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_filename(ticker: str) -> str:
    """Strip '^' so benchmark/VIX tickers map to safe filenames."""
    return ticker.replace("^", "")


def read_raw_ticker(
    ticker: str, raw_dir: str | Path, cutoff: str
) -> pd.DataFrame:
    """Read one raw Parquet file; defensively enforce date <= cutoff."""
    path = Path(raw_dir) / (sanitize_filename(ticker) + ".parquet")
    df = pd.read_parquet(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = pd.to_datetime(cutoff)
    df = df[df["date"] <= end].sort_values("date").reset_index(drop=True)
    return df


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    """Save a matplotlib figure and close it (no interactive display)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot builders (descriptive only)
# ---------------------------------------------------------------------------

def plot_adjusted_price(dates, adj, ticker, path) -> Path:
    """Historical Adjusted Close line plot for one ticker."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, adj, linewidth=1.2)
    ax.set_title(f"{ticker} — Adjusted Close (descriptive history)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    fig.autofmt_xdate()
    return save_fig(fig, path)


def plot_moving_averages(dates, adj, ticker, path) -> Path:
    """Adjusted Close with trailing 20/50/200-day SMAs (EDA indicators)."""
    s = pd.Series(np.asarray(adj, dtype=float))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, s.values, label="Adj Close", linewidth=1.2)
    for w in SMA_WINDOWS:
        sma = s.rolling(window=w, min_periods=w).mean()
        ax.plot(dates, sma.values, label=f"{w}-day SMA", linewidth=1.0)
    ax.set_title(f"{ticker} — Adjusted Close with trailing SMAs (EDA only)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    return save_fig(fig, path)


def plot_rolling_volatility(dates, ann_vol, ticker, path) -> Path:
    """Annualized 20-day rolling volatility line (trailing window)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, ann_vol, linewidth=1.2)
    ax.set_title(f"{ticker} — 20-day rolling annualized volatility (trailing)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (decimal)")
    fig.autofmt_xdate()
    return save_fig(fig, path)


def plot_return_distribution(log_returns, ticker, path) -> Path:
    """Histogram of daily log returns for one ticker."""
    r = pd.to_numeric(log_returns, errors="coerce").dropna().values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r, bins=80)
    ax.set_title(f"{ticker} — daily log-return distribution (descriptive)")
    ax.set_xlabel("Daily log return (adj close)")
    ax.set_ylabel("Frequency")
    return save_fig(fig, path)


def plot_return_qq(log_returns, ticker, path) -> Path:
    """Normal QQ plot of daily log returns (tail inspection)."""
    r = pd.to_numeric(log_returns, errors="coerce").dropna().values
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(r, dist="norm")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(osm[0], osm[1], s=4)
    line = slope * np.asarray(osm[0]) + intercept
    ax.plot(osm[0], line, linewidth=1.2)
    ax.set_title(f"{ticker} — log-return QQ vs normal (tail check)")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    return save_fig(fig, path)


def plot_normalized_equity_prices(dates, normalized: pd.DataFrame, path) -> Path:
    """All equities rebased to 100 at first obs (descriptive normalization)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in normalized.columns:
        ax.plot(dates, normalized[col].values, label=col, linewidth=1.0)
    ax.set_title("Equities rebased to 100 at first observation (descriptive)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased price (first adj close = 100)")
    ax.legend(ncol=2, fontsize="small")
    fig.autofmt_xdate()
    return save_fig(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, path) -> Path:
    """Heatmap of the equity log-return correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=7)
    ax.set_title("Equity daily log-return correlation (aligned dates)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    return save_fig(fig, path)


def plot_gspc_price(dates, adj, path) -> Path:
    """S&P 500 Adjusted Close history (market context)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, adj, linewidth=1.2)
    ax.set_title("S&P 500 (^GSPC) — Adjusted Close (market context)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close")
    fig.autofmt_xdate()
    return save_fig(fig, path)


def plot_vix(dates, close, path) -> Path:
    """VIX history (market context; implied volatility, percent)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, close, linewidth=1.2)
    ax.set_title("VIX (^VIX) — Close history (implied vol, percent)")
    ax.set_xlabel("Date")
    ax.set_ylabel("VIX level")
    fig.autofmt_xdate()
    return save_fig(fig, path)


def plot_realized_vs_vix(dates, realized_pct, vix, path) -> Path:
    """S&P 500 20-day realized vol (%, trailing) vs VIX (different measures)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, realized_pct, label="GSPC 20d realized vol, ann. (%)",
            linewidth=1.2)
    ax.plot(dates, vix, label="VIX close (implied, %)", linewidth=1.0)
    ax.set_title("GSPC realized volatility vs VIX (different measures; context only)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.autofmt_xdate()
    return save_fig(fig, path)


# ---------------------------------------------------------------------------
# Main EDA pipeline
# ---------------------------------------------------------------------------

def run_eda(
    config_path: str | Path = "config.yaml",
    raw_dir: str | Path = RAW_DIR,
    tables_dir: str | Path = TABLES_DIR,
    plots_dir: str | Path = PLOTS_DIR,
) -> dict:
    """Run the full descriptive EDA: read raw, write tables + plots.

    Returns a summary dict with output counts. Never writes to raw_dir.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(config_path)
    universe = cfg["universe"]
    equities: list[str] = list(universe["tickers"])
    benchmark: str = universe["benchmark"]
    risk_ticker: str = universe["risk_regime_ticker"]
    cutoff: str = universe["research_cutoff_date"]

    tables_dir = Path(tables_dir)
    plots_dir = Path(plots_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = equities + [benchmark, risk_ticker]
    frames = {t: read_raw_ticker(t, raw_dir, cutoff) for t in all_tickers}

    # 1. Data quality (all tickers incl. benchmark + VIX)
    quality = pd.DataFrame(
        [data_quality_summary(t, frames[t]) for t in all_tickers]
    )
    quality.to_csv(tables_dir / "data_quality.csv", index=False)

    # Per-equity returns (adjusted close, descriptive)
    logrets = {
        t: compute_log_returns(frames[t]["adj_close"]) for t in equities
    }

    # 2. Price descriptives (close / adj_close / volume)
    desc_rows = []
    for t in equities:
        d = frames[t][["close", "adj_close", "volume"]].describe(
            percentiles=[0.25, 0.5, 0.75]
        )
        for col in ["close", "adj_close", "volume"]:
            row = {"ticker": t, "field": col}
            for stat in ["count", "mean", "std", "min", "25%", "50%",
                         "75%", "max"]:
                row[stat] = float(d.loc[stat, col])
            row["median"] = row["50%"]
            desc_rows.append(row)
    pd.DataFrame(desc_rows).to_csv(
        tables_dir / "price_descriptive_stats.csv", index=False
    )

    # 3+9. Return stats with tail percentiles
    ret_stats = pd.DataFrame(
        [return_statistics(t, logrets[t]) for t in equities]
    )
    ret_stats.to_csv(tables_dir / "return_statistics.csv", index=False)

    for t in equities:
        dates = frames[t]["date"]
        adj = frames[t]["adj_close"]
        r = logrets[t]
        _, ann = rolling_realized_vol(r)
        plot_adjusted_price(dates, adj, t, plots_dir / f"{t}_adjusted_price.png")
        plot_moving_averages(dates, adj, t,
                             plots_dir / f"{t}_moving_averages.png")
        plot_rolling_volatility(dates, ann, t,
                                plots_dir / f"{t}_rolling_volatility.png")
        plot_return_distribution(r, t,
                                 plots_dir / f"{t}_return_distribution.png")
        plot_return_qq(r, t, plots_dir / f"{t}_return_qq.png")

    # 4b. Combined normalized comparison (equities only, date-aligned)
    norm = align_series_by_date(
        {t: normalize_to_100(frames[t]["adj_close"]) for t in equities},
        {t: frames[t]["date"] for t in equities},
    )
    plot_normalized_equity_prices(
        norm.index, norm,
        plots_dir / "normalized_equity_prices.png",
    )

    # 7. VIX / market context
    gspc = frames[benchmark]
    vix = frames[risk_ticker]
    gspc_ret = compute_log_returns(gspc["adj_close"])
    _, gspc_ann = rolling_realized_vol(gspc_ret)
    plot_gspc_price(gspc["date"], gspc["adj_close"],
                    plots_dir / "GSPC_price.png")
    plot_vix(vix["date"], vix["close"], plots_dir / "VIX.png")
    overlap = pd.DataFrame({
        "date": gspc["date"].values,
        "realized_pct": (gspc_ann.values * 100.0),
    }).dropna()
    vix_c = pd.DataFrame(
        {"date": vix["date"].values, "vix": vix["close"].values}
    ).dropna()
    both = overlap.merge(vix_c, on="date", how="inner")
    plot_realized_vs_vix(both["date"], both["realized_pct"], both["vix"],
                         plots_dir / "GSPC_realized_vs_vix.png")

    # 8. Correlation of equity log returns on genuinely matched dates
    wide = align_series_by_date(
        logrets, {t: frames[t]["date"] for t in equities}
    )
    wide = wide.dropna(how="any")  # aligned dates only; no filling
    corr = correlation_matrix(wide)
    corr.to_csv(tables_dir / "correlation_matrix.csv")
    plot_correlation_heatmap(corr, plots_dir / "equity_return_correlation.png")

    # 10. ADF on adj_close and log returns (equities)
    adf_rows = []
    for t in equities:
        adf_rows.append(adf_summary(t, "adj_close", frames[t]["adj_close"]))
        adf_rows.append(adf_summary(t, "log_return", logrets[t]))
    pd.DataFrame(adf_rows).to_csv(tables_dir / "adf_results.csv", index=False)

    # 11. Calendar descriptives (equities; with sample sizes)
    cal_frames = []
    for t in equities:
        c = calendar_return_stats(frames[t]["date"], logrets[t])
        c.insert(0, "ticker", t)
        cal_frames.append(c)
    pd.concat(cal_frames, ignore_index=True).to_csv(
        tables_dir / "calendar_return_stats.csv", index=False
    )

    # 12. Cross-asset summary (equities)
    cross = pd.DataFrame(
        [cross_asset_row(t, frames[t]["date"], frames[t]["adj_close"],
                         logrets[t]) for t in equities]
    )
    cross.to_csv(tables_dir / "cross_asset_summary.csv", index=False)

    n_tables = len(list(tables_dir.glob("*.csv")))
    n_plots = len(list(plots_dir.glob("*.png")))
    summary = {
        "tickers": all_tickers,
        "n_tables": n_tables,
        "n_plots": n_plots,
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
    }
    print("=" * 72)
    print("EDA COMPLETE (descriptive only; no modelling)")
    print(f"Tables: {n_tables} -> {tables_dir}")
    print(f"Plots : {n_plots} -> {plots_dir}")
    print(f"Tickers covered: {', '.join(all_tickers)}")
    print("=" * 72)
    return summary


def main(config_path: str | Path = "config.yaml") -> dict:
    """CLI entry point: ``python -m src.eda``."""
    return run_eda(config_path=config_path)


if __name__ == "__main__":
    main()
