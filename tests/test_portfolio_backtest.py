"""Tests for src.portfolio_backtest (Task 4D).

Leakage tests (A-G) on deterministic synthetic panels are the core:
future shocks must never move historical weights. Real-data CLI output
is validated structurally, never for profitability.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import portfolio_math as pm
from src import portfolio_optimizer as popt
from src.portfolio_backtest import (
    BacktestResult,
    DEFAULT_SOLVER_OPTIONS,
    calmar_ratio,
    run_rolling_backtest,
)

TICKERS = ["A", "B", "C"]
LOOKBACK = 60


def make_panel(n=400, seed=0, shock=None):
    """Deterministic synthetic LOG-return panel with a regime change.

    shock: optional (start_date_str, scale) multiplying later returns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    base = rng.normal(0.0005, 0.01, (n, 3))
    base[n // 2:] += np.array([0.002, -0.001, 0.0])  # visible regime shift
    panel = pd.DataFrame(base, index=dates, columns=TICKERS)
    bench = pd.Series(rng.normal(0.0004, 0.008, n), index=dates)
    if shock is not None:
        start, scale = shock
        panel.loc[pd.Timestamp(start):] *= scale
    return panel, bench


def run(panel, bench, **kw):
    args = {"tickers": TICKERS, "lookback": LOOKBACK}
    args.update(kw)
    return run_rolling_backtest(panel, bench, **args)


def weights_of(result, strategy):
    """{rebalance_date: weight vector} for one strategy."""
    return {rec["rebalance_date"]: np.array(
        [rec["weights"][a] for a in TICKERS])
        for rec in result.rebalance_records
        if rec["strategy"] == strategy}


def test_backtest_runs_and_reports_period():
    panel, bench = make_panel()
    r = run(panel, bench)
    assert isinstance(r, BacktestResult)
    assert r.summary["n_rebalance_dates"] >= 3
    assert r.summary["n_failed_rebalances"] == 0
    assert set(r.metrics) == {"equal_weight", "rolling_min_variance",
                              "rolling_max_sharpe", "sp500"}
    for rec in r.rebalance_records:
        assert rec["status"] == "ok"


# 15A. Future-return perturbation leaves historical weights unchanged

def test_future_return_shock_does_not_move_past_weights():
    panel, bench = make_panel()
    base = run(panel, bench)
    dates = sorted(weights_of(base, "rolling_max_sharpe"))
    pivot = dates[1]  # shock everything strictly after 2nd rebalance
    shock_start = (pivot + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    panel2, _ = make_panel(shock=(shock_start, 8.0))
    # shock is real: tail returns actually changed
    assert not np.allclose(panel[shock_start:].to_numpy(),
                           panel2[shock_start:].to_numpy())
    shocked = run(panel2, bench)
    for strat in ("rolling_min_variance", "rolling_max_sharpe",
                  "equal_weight"):
        w_base, w_new = weights_of(base, strat), weights_of(shocked, strat)
        assert sorted(w_new) == sorted(w_base)
        for d in dates[:2]:  # at and before the pivot: identical
            assert np.allclose(w_base[d], w_new[d], atol=1e-12), (strat, d)


# 15B. Future-covariance perturbation leaves historical inputs unchanged

def test_future_covariance_shock_leaves_history_unchanged():
    panel, bench = make_panel()
    base = run(panel, bench)
    dates = sorted(weights_of(base, "rolling_min_variance"))
    pivot = dates[2]
    shock_start = (pivot + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
    panel2, _ = make_panel(shock=(shock_start, 0.05))
    shocked = run(panel2, bench)
    for strat in ("rolling_min_variance", "rolling_max_sharpe"):
        w_base, w_new = weights_of(base, strat), weights_of(shocked, strat)
        for d in dates[:3]:
            assert np.allclose(w_base[d], w_new[d], atol=1e-12), (strat, d)
    # estimation windows recorded before the pivot end before the shock
    for rec in base.rebalance_records:
        if rec["rebalance_date"] <= pivot:
            assert rec["estimation_end"] < pd.Timestamp(shock_start)


# 15C. Weights equal direct optimizer output on the preceding window only

def test_weights_match_window_only_optimization():
    panel, bench = make_panel()
    r = run(panel, bench, risk_free_rate_annual=0.02)
    rf_daily = 0.02 / 252
    checked = 0
    for rec in r.rebalance_records:
        if rec["strategy"] != "rolling_max_sharpe":
            continue
        window = panel.loc[rec["estimation_start"]:rec["estimation_end"]]
        simple = np.expm1(window.to_numpy())
        mu = np.array([pm.mean_return(simple[:, j]) for j in range(3)])
        cov = pm.covariance_matrix(simple)
        expect = popt.maximum_sharpe_portfolio(
            mu, cov, risk_free_rate=rf_daily,
            solver_options=dict(DEFAULT_SOLVER_OPTIONS)).weights
        got = np.array([rec["weights"][a] for a in TICKERS])
        assert np.allclose(got, expect, atol=1e-8)
        checked += 1
    assert checked >= 3


# 15D. No future observation in any estimation window

def test_no_future_observation_in_windows():
    panel, bench = make_panel()
    r = run(panel, bench)
    for rec in r.rebalance_records:
        assert rec["estimation_end"] < rec["rebalance_date"]
        assert rec["estimation_start"] <= rec["estimation_end"]
        window = panel.loc[rec["estimation_start"]:rec["estimation_end"]]
        assert len(window) == LOOKBACK


# 15E. First rebalance only after the full lookback exists

def test_first_rebalance_requires_full_lookback():
    panel, bench = make_panel(n=300)
    r = run_rolling_backtest(panel, bench, tickers=TICKERS, lookback=252)
    first = min(rec["rebalance_date"] for rec in r.rebalance_records)
    n_prior = int((panel.index < first).sum())
    assert n_prior >= 252
    assert r.summary["n_rebalance_dates"] >= 1


# 15F. Benchmark isolation

def test_benchmark_isolated_from_weights():
    panel, bench = make_panel()
    bench = bench * 50 + 0.02  # wildly distinctive benchmark path
    r = run(panel, bench)
    assets = set()
    for rec in r.rebalance_records:
        assets.update(rec["weights"])
    assert "sp500" not in assets and "GSPC" not in assets
    assert assets == set(TICKERS)
    # strategy returns use only universe returns: recompute one day
    recs = [x for x in r.rebalance_records
            if x["strategy"] == "rolling_min_variance"]
    day = r.dates[5]
    w = np.array([recs[0]["weights"][a] for a in TICKERS])
    simple_day = np.expm1(panel.loc[day, TICKERS].to_numpy())
    assert r.daily_returns["rolling_min_variance"][5] == pytest.approx(
        float(w @ simple_day))


# 15G. Far-future shock cannot move any historical weight

def test_far_future_shock_moves_no_weights_but_moves_values():
    panel, bench = make_panel()
    base = run(panel, bench)
    last_reb = max(rec["rebalance_date"] for rec in base.rebalance_records)
    shock_start = (last_reb + pd.tseries.offsets.BDay(1))\
        .strftime("%Y-%m-%d")
    panel2, _ = make_panel(shock=(shock_start, 10.0))
    shocked = run(panel2, bench)
    for strat in ("rolling_min_variance", "rolling_max_sharpe"):
        w_base, w_new = weights_of(base, strat), weights_of(shocked, strat)
        for d in w_base:
            assert np.allclose(w_base[d], w_new[d], atol=1e-12)
    # ... while realized performance does reflect the shock
    assert not np.allclose(base.values["rolling_max_sharpe"],
                           shocked.values["rolling_max_sharpe"])


# 17. Determinism, accounting identities, failure policy, validation

def test_determinism():
    panel, bench = make_panel()
    a = run(panel, bench)
    b = run(panel, bench)
    assert np.array_equal(a.values["rolling_min_variance"],
                          b.values["rolling_min_variance"])
    assert [r["rebalance_date"] for r in a.rebalance_records] == \
           [r["rebalance_date"] for r in b.rebalance_records]


def test_value_curve_compounds_and_turnover_documented():
    panel, bench = make_panel()
    r = run(panel, bench, initial_capital=10_000.0)
    v, ret = r.values["equal_weight"], r.daily_returns["equal_weight"]
    assert v[0] == pytest.approx(10_000.0 * (1 + ret[0]))
    assert np.allclose(v[1:], v[:-1] * (1 + ret[1:]))
    eq_turn = [x["turnover"] for x in r.rebalance_records
               if x["strategy"] == "equal_weight"]
    assert eq_turn[0] == pytest.approx(1.0)  # initial purchase from cash
    assert np.allclose(eq_turn[1:], 0.0)  # constant 1/N targets: no trades
    for rec in r.rebalance_records:
        assert rec["turnover"] >= 0.0


def test_optimizer_failure_holds_cash_and_records_status():
    panel, bench = make_panel()
    r = run(panel, bench, solver_options={"maxiter": 0})  # force failure
    mv_recs = [x for x in r.rebalance_records
               if x["strategy"] == "rolling_min_variance"]
    assert all(x["status"].startswith("failed") for x in mv_recs)
    assert all(x["weights"] is None for x in mv_recs)  # never positioned
    # no position ever taken -> flat cash curve at initial capital
    assert np.all(r.values["rolling_min_variance"] == pytest.approx(100_000.0))
    assert r.summary["n_failed_rebalances"] > 0
    # equal weight is unaffected by optimizer failure
    assert all(x["status"] == "ok" for x in r.rebalance_records
               if x["strategy"] == "equal_weight")


def test_inputs_validated():
    panel, bench = make_panel()
    bad = panel.copy()
    bad.iloc[10, 0] = np.nan
    with pytest.raises(ValueError):
        run(bad, bench)
    with pytest.raises(ValueError):
        run_rolling_backtest(panel.iloc[:30], bench.iloc[:30],
                             tickers=TICKERS, lookback=LOOKBACK)
    with pytest.raises(ValueError):
        run(panel, bench, lookback=1)
    with pytest.raises(ValueError):
        run(panel, bench, var_confidence=1.5)
    dup = pd.concat([panel, panel.iloc[[5]]]).sort_index()
    with pytest.raises(ValueError):
        run(dup, bench)


def test_metrics_use_task4a_conventions():
    panel, bench = make_panel()
    r = run(panel, bench)
    m = r.metrics["rolling_min_variance"]
    v = r.values["rolling_min_variance"]
    assert m["maximum_drawdown"] == pytest.approx(pm.maximum_drawdown(v))
    assert m["maximum_drawdown"] <= 0.0
    assert m["annualized_volatility"] == pytest.approx(
        pm.annualized_volatility(pm.return_volatility(
            r.daily_returns["rolling_min_variance"])))
    assert m["sortino_ratio"] == pytest.approx(
        pm.sortino_ratio(r.daily_returns["rolling_min_variance"], 0.0))
    assert m["historical_var_daily"] == pytest.approx(
        pm.historical_var(r.daily_returns["rolling_min_variance"], 0.95))
    assert m["historical_var_daily"] >= 0.0  # positive-loss convention
    for k in ("cumulative_return", "annualized_return_geo", "sharpe_ratio",
              "calmar_ratio", "total_turnover"):
        assert math.isfinite(m[k]) or math.isnan(m[k])


def test_calmar_helper():
    assert calmar_ratio(0.10, -0.25) == pytest.approx(0.4)
    assert calmar_ratio(0.10, 0.0) == math.inf
    assert math.isnan(calmar_ratio(0.0, 0.0))
    assert math.isnan(calmar_ratio(-0.05, 0.0))
    with pytest.raises(ValueError):
        calmar_ratio(0.10, 0.25)  # positive drawdown violates convention
