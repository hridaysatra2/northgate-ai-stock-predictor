"""Raw market-data ingestion layer (yfinance) for Northgate AI Stock Predictor.

Scope: fetch daily OHLCV + Adjusted Close for the configured candidate
equities, the S&P 500 benchmark ("^GSPC") and the VIX risk-regime ticker
("^VIX"), normalize to a fixed schema, enforce the inclusive research
cutoff, and cache one Parquet file per ticker under ``data/raw/market/``.

Filename sanitization rule:
    The "^" character is not a valid/portable filename character for this
    project, so it is stripped from ticker symbols when building filenames.
    E.g. "^GSPC" -> "GSPC.parquet", "^VIX" -> "VIX.parquet".
    Implemented in :func:`sanitize_ticker_for_filename` via
    ``ticker.replace("^", "")``.

Duplicate-date rule:
    If the source returns duplicate dates for a ticker, rows are sorted by
    date and deduplicated with ``keep="last"`` (the last occurrence in
    source order wins). After this step ``date`` is unique.

Gap heuristic (availability reporting only, NOT cleaning):
    Calendar-day gaps between consecutive observations of more than
    ``SUSPICIOUS_GAP_DAYS = 5`` days are reported as "suspicious".
    Rationale: an ordinary weekend produces a 3-day calendar gap
    (Friday -> Monday); a weekend plus one US market holiday produces at
    most a 4-day gap. A gap strictly greater than 5 calendar days therefore
    suggests multiple missing trading sessions beyond weekends/holidays.
    Ordinary weekend gaps are never flagged. No exchange-calendar
    dependency is used; full trading-calendar-aware data-quality analysis
    belongs to the later cleaning phase. Gaps are reported only, never
    filled/interpolated.

Idempotency:
    Each ticker file is deterministically overwritten from the same source
    request (sorted, deduplicated, fixed schema/column order). Repeated runs
    therefore produce the same LOGICAL dataset (same columns/rows/values/
    ordering/dtypes). Byte-for-byte Parquet equality is NOT guaranteed
    (metadata/compression may differ) and must not be used for comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]

# Gaps strictly greater than this many calendar days are "suspicious".
SUSPICIOUS_GAP_DAYS = 5

# Fewer than this many rows (but > 0) counts as "near-empty".
NEAR_EMPTY_THRESHOLD = 30

DEFAULT_OUTPUT_DIR = Path("data/raw/market")


def sanitize_ticker_for_filename(ticker: str) -> str:
    """Strip '^' from a ticker so it is safe for filenames.

    "^GSPC" -> "GSPC", "^VIX" -> "VIX". Other tickers are unchanged.
    """
    return ticker.replace("^", "")


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Convert to tz-naive datetime64 series, preserving calendar dates."""
    s = pd.to_datetime(series, utc=True, errors="coerce")
    # s is tz-aware UTC at this point; drop tz -> tz-naive.
    try:
        s = s.dt.tz_convert(None)
    except Exception:  # pragma: no cover - defensive fallback
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    return s


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten possible MultiIndex columns from yfinance to a single level."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def normalize_yfinance_dataframe(
    raw: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    """Normalize a raw yfinance dataframe to the canonical output schema.

    Output columns (exact order): date, open, high, low, close, adj_close,
    volume. No filling/interpolation/cleaning is performed. Applies the
    inclusive research-cutoff filter (``date <= end_date``) and the
    ``keep="last"`` duplicate-date rule.
    """
    if raw is None or raw.empty:
        raise ValueError("empty dataframe from yfinance")

    df = _flatten_yfinance_columns(raw)

    # Bring the DatetimeIndex ("Date"/"Datetime") into a column if needed.
    index_names = [str(n).lower() for n in df.index.names if n is not None]
    if isinstance(df.index, pd.DatetimeIndex) or any(
        n in ("date", "datetime") for n in index_names
    ):
        df = df.reset_index()

    # Map source columns (case/space-insensitive) to canonical names.
    col_map: dict[str, str] = {}
    for col in list(df.columns):
        key = str(col).strip().lower().replace("_", " ")
        key = " ".join(key.split())
        if key == "date" or key == "datetime":
            col_map[col] = "date"
        elif key == "open":
            col_map[col] = "open"
        elif key == "high":
            col_map[col] = "high"
        elif key == "low":
            col_map[col] = "low"
        elif key == "close":
            col_map[col] = "close"
        elif key in ("adj close",):
            col_map[col] = "adj_close"
        elif key == "volume":
            col_map[col] = "volume"
    df = df.rename(columns=col_map)

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns after rename: {missing}")

    df["date"] = _to_naive_datetime(df["date"])
    df = df.dropna(subset=["date"])

    # Inclusive research-cutoff filter: date <= end_date (and >= start_date).
    cutoff = pd.to_datetime(end_date)
    start = pd.to_datetime(start_date)
    # Normalize cutoff/start to tz-naive for comparison.
    cutoff = pd.to_datetime(cutoff, utc=True).tz_convert(None)
    start = pd.to_datetime(start, utc=True).tz_convert(None)
    df = df[(df["date"] >= start) & (df["date"] <= cutoff)]

    # Deterministic ordering + duplicate rule: keep="last".
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    return df[OUTPUT_COLUMNS]


def detect_suspicious_gaps(dates: pd.Series) -> list[dict]:
    """Return gaps > SUSPICIOUS_GAP_DAYS calendar days (reporting only)."""
    if dates is None or len(dates) < 2:
        return []
    s = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    gaps: list[dict] = []
    for i in range(1, len(s)):
        delta_days = (s.iloc[i] - s.iloc[i - 1]).days
        if delta_days > SUSPICIOUS_GAP_DAYS:
            gaps.append(
                {
                    "gap_start": s.iloc[i - 1].date().isoformat(),
                    "gap_end": s.iloc[i].date().isoformat(),
                    "gap_days": int(delta_days),
                }
            )
    return gaps


def build_availability_report(
    ticker: str,
    start_date: str,
    end_date: str,
    df: pd.DataFrame | None,
    error: str | None = None,
) -> dict:
    """Build a structured availability summary for one ticker."""
    if df is None or df.empty:
        return {
            "ticker": ticker,
            "requested_start": start_date,
            "requested_cutoff": end_date,
            "first_date": None,
            "last_date": None,
            "n_observations": 0,
            "is_empty": True,
            "is_near_empty": True,
            "suspicious_gaps": [],
            "status": "failed" if error else "empty",
            "error": error,
        }
    gaps = detect_suspicious_gaps(df["date"])
    n = int(len(df))
    return {
        "ticker": ticker,
        "requested_start": start_date,
        "requested_cutoff": end_date,
        "first_date": pd.to_datetime(df["date"].iloc[0]).date().isoformat(),
        "last_date": pd.to_datetime(df["date"].iloc[-1]).date().isoformat(),
        "n_observations": n,
        "is_empty": False,
        "is_near_empty": bool(n < NEAR_EMPTY_THRESHOLD),
        "suspicious_gaps": gaps,
        "status": "ok",
        "error": error,
    }


def print_availability_summary(reports: list[dict]) -> None:
    """Print a human-readable availability report for all tickers."""
    print("=" * 78)
    print("DATA AVAILABILITY REPORT")
    print("=" * 78)
    for r in reports:
        print(f"ticker            : {r['ticker']}")
        print(f"  requested start : {r['requested_start']}")
        print(f"  requested cutoff: {r['requested_cutoff']}")
        print(f"  first obs       : {r['first_date']}")
        print(f"  last obs        : {r['last_date']}")
        print(f"  n_observations  : {r['n_observations']}")
        print(f"  status          : {r['status']}")
        print(f"  is_empty        : {r['is_empty']}")
        print(f"  is_near_empty   : {r['is_near_empty']}")
        if r.get("error"):
            print(f"  error           : {r['error']}")
        gaps = r.get("suspicious_gaps") or []
        if gaps:
            print(f"  suspicious gaps ({len(gaps)}):")
            for g in gaps:
                print(f"    {g['gap_start']} -> {g['gap_end']} ({g['gap_days']}d)")
        else:
            print("  suspicious gaps : none")
        print("-" * 78)


def fetch_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[dict]:
    """Fetch, normalize, cache and report raw market data for tickers.

    Each ticker is fetched individually with ``yfinance.download`` using
    ``auto_adjust=False`` so raw Close and Adjusted Close are preserved as
    separate fields. Results are filtered to ``date <= end_date``
    (inclusive cutoff), sorted, deduplicated (keep="last"), saved as
    ``<sanitized>.parquet`` under ``output_dir``, and summarized in an
    availability report. A failure for one ticker never stops the rest.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # yfinance `end` is exclusive: request one extra day, then apply the
    # inclusive `date <= end_date` filter inside normalization.
    cutoff_ts = pd.to_datetime(end_date)
    yf_end = (cutoff_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    reports: list[dict] = []
    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=yf_end,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - must not crash others
            logger.warning("fetch failed for %s: %s", ticker, exc)
            reports.append(
                build_availability_report(
                    ticker, start_date, end_date, None, error=str(exc)
                )
            )
            continue

        if raw is None or raw.empty:
            logger.warning("empty response for ticker %s", ticker)
            reports.append(
                build_availability_report(
                    ticker, start_date, end_date, None, error="empty response"
                )
            )
            continue

        try:
            df = normalize_yfinance_dataframe(raw, start_date, end_date)
        except Exception as exc:  # noqa: BLE001 - must not crash others
            logger.warning("normalization failed for %s: %s", ticker, exc)
            reports.append(
                build_availability_report(
                    ticker, start_date, end_date, None, error=str(exc)
                )
            )
            continue

        if df.empty:
            logger.warning("no rows within range for ticker %s", ticker)
            reports.append(
                build_availability_report(
                    ticker, start_date, end_date, None, error="no rows in range"
                )
            )
            continue

        fname = sanitize_ticker_for_filename(ticker) + ".parquet"
        df.to_parquet(out_dir / fname, index=False)
        reports.append(build_availability_report(ticker, start_date, end_date, df))

    return reports


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load ingestion configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str | Path = "config.yaml") -> list[dict]:
    """Entry point: load config, fetch all tickers, print availability."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(config_path)
    universe = cfg["universe"]
    tickers: list[str] = list(universe["tickers"])
    for extra in (universe.get("benchmark"), universe.get("risk_regime_ticker")):
        if extra and extra not in tickers:
            tickers.append(extra)
    start_date = universe["start_date"]
    cutoff = universe["research_cutoff_date"]
    reports = fetch_market_data(tickers, start_date, cutoff)
    print_availability_summary(reports)
    return reports


if __name__ == "__main__":
    main()
