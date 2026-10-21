"""Tests for src.eda helpers (synthetic data only; no internet)."""

import numpy as np
import pandas as pd
from pytest import approx

from src import eda
from src.eda import (
    TRADING_DAYS_PER_YEAR,
    adf_summary,
    compute_log_returns,
    correlation_matrix,
    data_quality_summary,
    maximum_drawdown,
    normalize_to_100,
    rolling_realized_vol,
)


# 1. log return calculation (uses adj_close ratio, first value NaN)
def test_1_log_returns():
    adj = pd.Series([100.0, 110.0, 121.0])
    r = compute_log_returns(adj)
    assert r.isna().iloc[0]
    assert r.iloc[1] == np.log(110.0 / 100.0)
    assert r.iloc[2] == np.log(121.0 / 110.0)


def test_1b_log_returns_rejects_raw_close_concept():
    # adj differs from raw close; helper must follow the series it is given.
    adj = pd.Series([100.0, 100.0 * 0.9])
    r = compute_log_returns(adj)
    assert r.iloc[1] == np.log(0.9)


# 2. normalized price calculation (first -> 100)
def test_2_normalized_price():
    s = pd.Series([50.0, 100.0, 75.0])
    n = normalize_to_100(s)
    assert n.tolist() == [100.0, 200.0, 150.0]


# 3. rolling volatility emits nothing before enough observations
def test_3_rolling_vol_min_periods():
    r = pd.Series(np.linspace(0.001, 0.025, 25))
    std, ann = rolling_realized_vol(r, window=20)
    assert std.iloc[:19].isna().all()
    assert std.iloc[19:].notna().all()
    assert ann.iloc[:19].isna().all()
    assert ann.iloc[19:].notna().all()


# 4. annualized volatility uses sqrt(252)
def test_4_annualized_uses_sqrt_252():
    r = pd.Series(np.linspace(-0.02, 0.02, 30))
    std, ann = rolling_realized_vol(r, window=20)
    expected = std * np.sqrt(252)
    pd.testing.assert_series_equal(ann, expected, check_names=False)
    assert TRADING_DAYS_PER_YEAR == 252
    assert np.sqrt(TRADING_DAYS_PER_YEAR) == np.sqrt(252)


# 5. correlation behaves correctly on a small known dataset
def test_5_correlation_known_data():
    a = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    wide = pd.DataFrame({
        "A": a,
        "B": a * 2.0,    # perfectly correlated with A
        "C": -a,         # perfectly anti-correlated with A
    })
    corr = correlation_matrix(wide)
    assert corr.loc["A", "B"] == 1.0
    assert corr.loc["A", "C"] == -1.0
    assert corr.loc["A", "A"] == 1.0
    assert corr.shape == (3, 3)


# 6. maximum drawdown calculation
def test_6_maximum_drawdown():
    # wealth: 1 -> 1.1 -> 0.88 -> 0.968 ; peak 1.1 ; trough dd = 0.88/1.1-1
    simple = pd.Series([0.0, 0.10, -0.20, 0.10])
    assert maximum_drawdown(simple) == approx(0.88 / 1.1 - 1.0)

    # monotone gains -> no drawdown
    assert maximum_drawdown(pd.Series([0.01, 0.02, 0.03])) == approx(0.0)


# 7. data-quality summary detects duplicates and missing values
def test_7_data_quality_detects_issues():
    df = pd.DataFrame({
        "date": pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-06"]
        ),
        "open": [1.0, np.nan, 3.0, 4.0],
        "high": [1.0, 2.0, 3.0, 4.0],
        "low": [1.0, 2.0, 3.0, 4.0],
        "close": [1.0, 2.0, 3.0, 4.0],
        "adj_close": [1.0, 2.0, 3.0, 4.0],
        "volume": [10, 20, 30, 40],
    })
    q = data_quality_summary("TEST", df)
    assert q["n_observations"] == 4
    assert q["n_duplicate_dates"] == 1
    assert q["missing_open"] == 1
    assert q["missing_close"] == 0
    assert q["first_date"] == "2020-01-02"
    assert q["last_date"] == "2020-01-06"


# 8. ADF output contains required fields (statistical fn mocked)
def test_8_adf_output_fields_mocked(monkeypatch):
    captured = {}

    def fake_adfuller(series, autolag="AIC", **kwargs):
        captured["n"] = len(series)
        return (-2.5, 0.12, 3, len(series),
                {"1%": -3.5, "5%": -2.9, "10%": -2.6}, None)

    monkeypatch.setattr(eda, "adfuller", fake_adfuller)
    out = adf_summary("MOCK", "log_return", pd.Series(range(50), dtype=float))
    assert out["ticker"] == "MOCK"
    assert out["series"] == "log_return"
    assert out["adf_statistic"] == -2.5
    assert out["p_value"] == 0.12
    assert out["n_observations"] == captured["n"]
    assert out["crit_1pct"] == -3.5
    assert out["crit_5pct"] == -2.9
    assert out["crit_10pct"] == -2.6
    # no boolean verdict imposed
    assert "stationary" not in out


def test_8b_adf_real_runs_offline():
    # Small synthetic stationary series; just checks the real wiring works.
    rng = np.random.default_rng(42)
    stationary = pd.Series(rng.normal(0, 1, 200))
    out = adf_summary("SYN", "log_return", stationary)
    for key in ("ticker", "series", "adf_statistic", "p_value",
                "n_observations", "crit_1pct", "crit_5pct", "crit_10pct"):
        assert key in out
    assert 0.0 <= out["p_value"] <= 1.0
