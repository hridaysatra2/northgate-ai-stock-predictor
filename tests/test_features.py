"""Tests for src.features (synthetic + local raw only; no internet)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
from pytest import approx

from src import features
from src.features import (
    EQUITY_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_market_features,
    build_own_features,
    build_supervised_frame,
    build_target,
    run_features,
)

CUTOFF = "2026-08-31"
RAW_DIR = Path("data/raw/market")


def make_raw(n=100, start="2020-01-01", seed=7, price0=100.0):
    """Synthetic raw-style frame (business days, positive prices)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(0.001, 0.01, n)
    adj = price0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "date": dates,
        "open": adj * 0.999,
        "high": adj * 1.001,
        "low": adj * 0.998,
        "close": adj,
        "adj_close": adj,
        "volume": (1000 + np.arange(n) * 10).astype(float),
    })


def make_market_frames(dates):
    """Tiny GSPC/VIX-style raw frames aligned to given dates."""
    n = len(dates)
    sp = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": 4000.0, "high": 4010.0, "low": 3990.0,
        "close": 4000.0 + np.arange(n),
        "adj_close": 4000.0 + np.arange(n),
        "volume": 1e9,
    })
    vx = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": 20.0, "high": 21.0, "low": 19.0,
        "close": 20.0 + 0.1 * np.arange(n),
        "adj_close": 20.0 + 0.1 * np.arange(n),
        "volume": 0.0,
    })
    return build_market_features(sp, vx)


def supervised(raw, cutoff=CUTOFF, **kw):
    sp_m, vx_m = make_market_frames(raw["date"])
    return build_supervised_frame("SYN", raw, sp_m, vx_m, cutoff, **kw)


# 1. Target correctness on known prices
def test_1_target_correctness():
    adj = pd.Series([100.0, 110.0, 121.0])
    t = build_target(adj)
    assert t.iloc[0] == np.log(110.0 / 100.0)
    assert t.iloc[1] == np.log(121.0 / 110.0)
    assert np.isnan(t.iloc[2])  # no t+1 available


# 2. Target alignment: row t uses NEXT observation, not current
def test_2_target_alignment():
    raw = make_raw(120)
    frame = supervised(raw)
    adj = raw["adj_close"].values
    # first supervised row: find its position in raw by date
    first_date = frame["date"].iloc[0]
    i = raw.index[raw["date"] == first_date][0]
    assert frame[TARGET_COLUMN].iloc[0] == np.log(adj[i + 1] / adj[i])
    # and the contemporaneous 1-day return uses the PREVIOUS pair instead
    assert frame["log_return_1d"].iloc[0] == np.log(adj[i] / adj[i - 1])


# 3. No future leakage: shocking future obs leaves earlier features unchanged
def test_3_no_future_leakage():
    raw = make_raw(120)
    base = supervised(raw)
    shocked = raw.copy()
    shocked.loc[shocked.index[80:], "adj_close"] *= 5.0
    shocked.loc[shocked.index[80:], "close"] *= 5.0
    after = supervised(shocked)
    feat_cols = [c for c in EQUITY_FEATURE_COLUMNS]
    shock_date = raw["date"].iloc[80]
    b = base[base["date"] < shock_date].reset_index(drop=True)
    a = after[after["date"] < shock_date].reset_index(drop=True)
    assert len(b) == len(a) and len(b) > 0
    pdt.assert_frame_equal(a[["date", "ticker"] + feat_cols],
                           b[["date", "ticker"] + feat_cols])
    # sanity: something AFTER the shock must actually differ (shock is real)
    assert not after.equals(base)


# 4. Rolling-window correctness vs manual calculation
def test_4_rolling_correctness():
    adj = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    vol = pd.Series([100.0] * 7)
    dates = pd.bdate_range("2020-01-01", periods=7)
    f = build_own_features(dates, adj, vol)
    assert f["sma_5"].iloc[4] == np.mean([10, 11, 12, 13, 14])
    assert np.isnan(f["sma_5"].iloc[3])
    assert f["momentum_5d"].iloc[5] == 15.0 / 10.0 - 1.0
    assert np.isnan(f["momentum_5d"].iloc[4])
    lr = np.log(adj / adj.shift(1))
    # NB: pandas rolling-std uses an online algorithm; allow 1-ULP tolerance.
    assert f["vol_5d"].iloc[5] == approx(lr.iloc[1:6].std(ddof=1), rel=1e-12)
    assert np.isnan(f["vol_5d"].iloc[4])


# 5. Annualization uses sqrt(252)
def test_5_annualization():
    raw = make_raw(90)
    f = build_own_features(raw["date"], raw["adj_close"], raw["volume"])
    for w in (5, 20, 60):
        pd.testing.assert_series_equal(
            f[f"vol_{w}d_ann"], f[f"vol_{w}d"] * np.sqrt(252),
            check_names=False)
    assert features.SQRT_252 == np.sqrt(252)
    assert features.TRADING_DAYS_PER_YEAR == 252


# 6. Volume ratio = volume / rolling mean volume
def test_6_volume_ratio():
    raw = make_raw(40)
    f = build_own_features(raw["date"], raw["adj_close"], raw["volume"])
    expected_mean = raw["volume"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(
        f["volume_sma_20"], expected_mean, check_names=False)
    expected_ratio = raw["volume"] / expected_mean
    pd.testing.assert_series_equal(
        f["volume_ratio_20"], expected_ratio.replace(
            [np.inf, -np.inf], np.nan), check_names=False)
    assert f["volume_change"].iloc[5] == (
        raw["volume"].iloc[5] / raw["volume"].iloc[4] - 1.0)


# 7. Market alignment by date; no synthetic market dates
def test_7_market_alignment():
    raw = make_raw(40)
    sp_m, vx_m = make_market_frames(raw["date"])
    # remove one market date -> merged features must be NaN there, not filled
    drop = raw["date"].iloc[25]
    sp_m2 = sp_m[sp_m["date"] != drop].reset_index(drop=True)
    frame = build_supervised_frame("SYN", raw, sp_m2, vx_m, CUTOFF)
    assert drop not in set(frame["date"])  # row dropped, never filled
    assert set(frame["date"]).issubset(set(raw["date"]))  # no new dates


# 8. Cutoff: no row beyond 2026-08-31
def test_8_cutoff():
    raw = make_raw(300, start="2026-01-01")
    # extend synthetic data past the cutoff deliberately
    extra = pd.DataFrame({
        "date": pd.bdate_range("2026-09-01", periods=10),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "adj_close": 1.0, "volume": 500.0,
    })
    raw = pd.concat([raw, extra], ignore_index=True)
    frame = supervised(raw, cutoff=CUTOFF)
    assert (frame["date"] <= pd.to_datetime(CUTOFF)).all()


# 9. Raw-data integrity: feature run never modifies data/raw
def test_9_raw_integrity(tmp_path):
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(RAW_DIR.glob("*.parquet"))}
    assert len(before) == 12
    run_features(raw_dir=RAW_DIR, output_dir=tmp_path)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(RAW_DIR.glob("*.parquet"))}
    assert before == after


# 10. Determinism: two runs -> identical logical frames
def test_10_determinism(tmp_path):
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    run_features(raw_dir=RAW_DIR, output_dir=out1)
    run_features(raw_dir=RAW_DIR, output_dir=out2)
    for p in sorted(out1.glob("*.parquet")):
        a = pd.read_parquet(p)
        b = pd.read_parquet(out2 / p.name)
        pdt.assert_frame_equal(a, b, check_dtype=True)


# 11. Schema: expected columns exist and dates are sorted
def test_11_schema(tmp_path):
    run_features(raw_dir=RAW_DIR, output_dir=tmp_path)
    expected = (["date", "ticker"] + EQUITY_FEATURE_COLUMNS
                + [TARGET_COLUMN])
    for p in sorted(tmp_path.glob("*.parquet")):
        df = pd.read_parquet(p)
        if p.stem == "VIX":
            assert TARGET_COLUMN not in df.columns  # diagnostics only
            assert "volume_change" not in df.columns  # no VIX volume
        else:
            assert list(df.columns) == expected
        assert df["date"].is_monotonic_increasing
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


# 12. Warm-up handling: long windows start NaN; supervised output is clean
def test_12_warmup():
    raw = make_raw(10)
    f = build_own_features(raw["date"], raw["adj_close"], raw["volume"])
    assert f["sma_50"].isna().all()
    assert f["vol_60d"].isna().all()
    assert f["sma_5"].iloc[:4].isna().all()
    assert f["sma_5"].iloc[4:].notna().all()
    frame = supervised(make_raw(120))
    required = EQUITY_FEATURE_COLUMNS + [TARGET_COLUMN]
    assert not frame[required].isna().any().any()
    assert len(frame) < 120  # warm-up + last-row target drop removed rows
