"""Tests for src.ingest (mocked yfinance, no network calls)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt

from src import ingest
from src.ingest import (
    OUTPUT_COLUMNS,
    fetch_market_data,
    sanitize_ticker_for_filename,
)

START = "2016-01-01"
CUTOFF = "2026-08-31"


def make_yf_frame(dates, close_start=100.0, adj_factor=0.9):
    """Build a mocked yfinance.download frame (flat columns, Date index).

    close and adj_close are deliberately different (adj = close*factor) so
    the mapping of the two fields is explicitly tested.
    """
    dates = list(pd.to_datetime(dates))
    n = len(dates)
    closes = [close_start + i for i in range(n)]
    df = pd.DataFrame(
        {
            "Open": [c - 1.0 for c in closes],
            "High": [c + 1.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": closes,
            "Adj Close": [c * adj_factor for c in closes],
            "Volume": [1000 + 10 * i for i in range(n)],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


def make_yf_multiindex_frame(dates):
    """Variant with MultiIndex columns like some yfinance versions."""
    flat = make_yf_frame(dates)
    flat.columns = pd.MultiIndex.from_product([list(flat.columns), ["AAPL"]])
    return flat


def read_parquet(path):
    return pd.read_parquet(path)


# TEST 1 — schema and sorting
def test_1_schema_and_sorting(tmp_path):
    dates = ["2026-08-28", "2026-08-26", "2026-08-27"]  # deliberately unsorted
    frame = make_yf_frame(dates)
    with patch.object(ingest.yf, "download", return_value=frame) as m:
        reports = fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
    out = tmp_path / "AAPL.parquet"
    assert out.exists()
    df = read_parquet(out)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing
    assert not df["date"].duplicated().any()
    assert reports[0]["status"] == "ok"
    assert m.called


# TEST 2 — duplicate handling (keep="last")
def test_2_duplicate_handling(tmp_path):
    dates = ["2026-08-26", "2026-08-27", "2026-08-27"]  # duplicate last date
    frame = make_yf_frame(dates)
    # Make the LAST duplicate row distinguishable: close == 999
    frame.iloc[2, frame.columns.get_loc("Close")] = 999.0
    frame.iloc[2, frame.columns.get_loc("Adj Close")] = 111.0
    with patch.object(ingest.yf, "download", return_value=frame):
        fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
    df = read_parquet(tmp_path / "AAPL.parquet")
    assert not df["date"].duplicated().any()
    assert len(df) == 2
    last = df[df["date"] == pd.to_datetime("2026-08-27")].iloc[0]
    assert last["close"] == 999.0  # keep="last" documented rule
    assert last["adj_close"] == 111.0


# TEST 3 — idempotency (logical dataframe equality, not byte equality)
def test_3_idempotency(tmp_path):
    frame = make_yf_frame(["2026-08-26", "2026-08-27", "2026-08-28"])
    with patch.object(ingest.yf, "download", return_value=frame):
        fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
        df1 = read_parquet(tmp_path / "AAPL.parquet")
        fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
        df2 = read_parquet(tmp_path / "AAPL.parquet")
    assert list(df1.columns) == list(df2.columns)
    pdt.assert_frame_equal(df1, df2, check_dtype=True)


# TEST 4 — failed ticker does not crash remaining tickers
def test_4_failed_ticker(tmp_path, caplog):
    good = make_yf_frame(["2026-08-26", "2026-08-27"])

    def side_effect(ticker, *args, **kwargs):
        if ticker == "BAD":
            raise RuntimeError("boom")
        if ticker == "EMPTY":
            return pd.DataFrame()
        return good

    import logging

    with patch.object(ingest.yf, "download", side_effect=side_effect):
        with caplog.at_level(logging.WARNING):
            reports = fetch_market_data(
                ["BAD", "EMPTY", "AAPL"], START, CUTOFF, output_dir=tmp_path
            )
    by_ticker = {r["ticker"]: r for r in reports}
    assert by_ticker["BAD"]["status"] == "failed"
    assert by_ticker["EMPTY"]["status"] in ("failed", "empty")
    assert by_ticker["AAPL"]["status"] == "ok"
    assert (tmp_path / "AAPL.parquet").exists()
    # warning/error was logged for the failures
    assert any(
        "BAD" in r.message or "EMPTY" in r.message or "empty" in r.message.lower() or "failed" in r.message.lower()
        for r in caplog.records
    )


# TEST 5 — benchmark and VIX use same code path, sanitized filenames
def test_5_benchmark_and_vix_filenames(tmp_path):
    frame = make_yf_frame(["2026-08-26", "2026-08-27"])
    with patch.object(ingest.yf, "download", return_value=frame):
        reports = fetch_market_data(
            ["^GSPC", "^VIX"], START, CUTOFF, output_dir=tmp_path
        )
    assert sanitize_ticker_for_filename("^GSPC") == "GSPC"
    assert sanitize_ticker_for_filename("^VIX") == "VIX"
    assert (tmp_path / "GSPC.parquet").exists()
    assert (tmp_path / "VIX.parquet").exists()
    assert not (tmp_path / "^GSPC.parquet").exists()
    assert not (tmp_path / "^VIX.parquet").exists()
    assert all(r["status"] == "ok" for r in reports)
    for p in (tmp_path / "GSPC.parquet", tmp_path / "VIX.parquet"):
        assert list(read_parquet(p).columns) == OUTPUT_COLUMNS


# TEST 6 — auto_adjust=False and close != adj_close preserved
def test_6_auto_adjust_false_and_close_mapping(tmp_path):
    frame = make_yf_frame(["2026-08-26", "2026-08-27"], close_start=200.0, adj_factor=0.5)
    with patch.object(ingest.yf, "download", return_value=frame) as m:
        fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
    # every call must explicitly pass auto_adjust=False
    for call in m.call_args_list:
        assert "auto_adjust" in call.kwargs
        assert call.kwargs["auto_adjust"] is False
    df = read_parquet(tmp_path / "AAPL.parquet")
    # mocked source had close != adj_close; both must survive separately.
    # NOTE: this does NOT claim real data always differs.
    assert (df["close"] != df["adj_close"]).all()
    assert df["close"].tolist() == [200.0, 201.0]
    assert df["adj_close"].tolist() == [100.0, 100.5]


# TEST 6b — MultiIndex yfinance columns also map correctly
def test_6b_multiindex_columns(tmp_path):
    frame = make_yf_multiindex_frame(["2026-08-26", "2026-08-27"])
    with patch.object(ingest.yf, "download", return_value=frame):
        fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
    df = read_parquet(tmp_path / "AAPL.parquet")
    assert list(df.columns) == OUTPUT_COLUMNS
    assert (df["close"] != df["adj_close"]).all()


# TEST 7 — research cutoff is inclusive (2026-08-31 kept, later dropped)
def test_7_research_cutoff_inclusive(tmp_path):
    dates = ["2026-08-29", "2026-08-30", "2026-08-31", "2026-09-01", "2026-09-02"]
    frame = make_yf_frame(dates)
    with patch.object(ingest.yf, "download", return_value=frame):
        fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
    df = read_parquet(tmp_path / "AAPL.parquet")
    assert (df["date"] <= pd.to_datetime(CUTOFF)).all()
    assert pd.to_datetime("2026-08-31") in set(df["date"])
    assert df["date"].max() == pd.to_datetime("2026-08-31")


# TEST 8 — availability report + gap heuristic
def test_8_availability_report_gaps(tmp_path, capsys):
    # Ordinary weekend gap (Fri 2026-08-07 -> Mon 2026-08-10 = 3 calendar days,
    # must NOT be flagged) plus one deliberately suspicious 10-day gap
    # (2026-08-10 -> 2026-08-20). All dates are within the research cutoff.
    dates = [
        "2026-08-03",  # Mon
        "2026-08-04",
        "2026-08-05",
        "2026-08-06",
        "2026-08-07",  # Fri
        "2026-08-10",  # Mon (normal weekend gap, must NOT be flagged)
        "2026-08-20",  # 10 calendar days later (suspicious)
    ]
    frame = make_yf_frame(dates)
    with patch.object(ingest.yf, "download", return_value=frame):
        reports = fetch_market_data(["AAPL"], START, CUTOFF, output_dir=tmp_path)
    r = reports[0]
    df = read_parquet(tmp_path / "AAPL.parquet")
    assert r["n_observations"] == 7 == len(df)
    assert r["first_date"] == "2026-08-03"
    assert r["last_date"] == "2026-08-20"
    assert r["status"] == "ok"
    gaps = r["suspicious_gaps"]
    # weekend gap (3d) must not be flagged; the 10d gap must be flagged.
    assert all(g["gap_days"] <= 10 for g in gaps)
    assert any(g["gap_days"] == 10 for g in gaps)
    assert not any(g["gap_days"] == 3 for g in gaps)
    # Heuristic documentation: >5 calendar days is suspicious.
    assert ingest.SUSPICIOUS_GAP_DAYS == 5
