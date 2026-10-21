"""Tests for src.portfolio_frontier (Task 4C).

Deterministic; numerical inputs only. No market data, no dates, no
backtesting. Optimizer outputs compared with tolerances, never exact
equality.
"""

import math
import time

import numpy as np
import pytest

from src import portfolio_math as pm
from src import portfolio_optimizer as popt
from src.portfolio_frontier import (
    DEFAULT_N_PORTFOLIOS,
    DEFAULT_SEED,
    TARGET_TOL,
    FrontierResult,
    MonteCarloResult,
    efficient_frontier,
    key_portfolios,
    simulate_portfolios,
)


def two_asset():
    mu = np.array([0.10, 0.06])
    cov = np.array([[0.04, 0.004], [0.004, 0.01]])
    return mu, cov


def three_asset():
    mu = np.array([0.12, 0.08, 0.05])
    cov = np.array([[0.04, 0.006, 0.002],
                    [0.006, 0.0225, 0.001],
                    [0.002, 0.001, 0.01]])
    return mu, cov


# ---------------------------------------------------------------------------
# 10A/B. MC constraints + shape
# ---------------------------------------------------------------------------

def test_mc_weight_constraints():
    mu, cov = three_asset()
    mc = simulate_portfolios(mu, cov, n_portfolios=500, seed=7)
    assert mc.weights.shape == (500, 3)
    assert np.all(mc.weights >= 0.0)
    assert np.allclose(mc.weights.sum(axis=1), 1.0, atol=1e-12)


def test_mc_default_size_and_seed():
    mu, cov = two_asset()
    mc = simulate_portfolios(mu, cov)
    assert mc.seed == DEFAULT_SEED == 42
    assert mc.n_portfolios == DEFAULT_N_PORTFOLIOS == 20_000
    assert mc.weights.shape == (20_000, 2)
    assert mc.n_assets == 2


def test_mc_dataframe_columns():
    mu, cov = two_asset()
    mc = simulate_portfolios(mu, cov, n_portfolios=10, seed=1,
                             asset_names=["AAA", "BBB"])
    df = mc.to_dataframe()
    assert list(df.columns) == ["portfolio_id", "w_AAA", "w_BBB",
                                "expected_return", "variance",
                                "volatility", "sharpe_ratio"]
    assert len(df) == 10


# ---------------------------------------------------------------------------
# 10C/D. Determinism + seed sensitivity
# ---------------------------------------------------------------------------

def test_mc_determinism_same_seed():
    mu, cov = three_asset()
    a = simulate_portfolios(mu, cov, n_portfolios=300, seed=42)
    b = simulate_portfolios(mu, cov, n_portfolios=300, seed=42)
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.expected_returns, b.expected_returns)
    assert np.array_equal(a.variances, b.variances)


def test_mc_different_seeds_differ():
    mu, cov = three_asset()
    a = simulate_portfolios(mu, cov, n_portfolios=300, seed=42)
    b = simulate_portfolios(mu, cov, n_portfolios=300, seed=43)
    assert not np.allclose(a.weights, b.weights)


# ---------------------------------------------------------------------------
# 10E. Stored metrics match independent Task 4A recomputation
# ---------------------------------------------------------------------------

def test_mc_metrics_match_portfolio_math():
    mu, cov = three_asset()
    rf = 0.02
    mc = simulate_portfolios(mu, cov, risk_free_rate=rf,
                             n_portfolios=500, seed=11)
    for i in (0, 7, 123, 499):
        w = mc.weights[i]
        assert mc.expected_returns[i] == pytest.approx(
            pm.portfolio_expected_return(w, mu))
        assert mc.variances[i] == pytest.approx(
            pm.portfolio_variance(w, cov))
        assert mc.volatilities[i] == pytest.approx(
            pm.portfolio_volatility(w, cov))
        assert mc.sharpe_ratios[i] == pytest.approx(
            pm.sharpe_ratio(mc.expected_returns[i], rf,
                            mc.volatilities[i]))


# ---------------------------------------------------------------------------
# 10F/G. Single asset + two-asset simplex
# ---------------------------------------------------------------------------

def test_mc_single_asset_all_ones():
    mc = simulate_portfolios(np.array([0.07]), np.array([[0.0225]]),
                             n_portfolios=100, seed=3)
    assert mc.weights.shape == (100, 1)
    assert np.all(mc.weights == 1.0)
    assert np.all(mc.expected_returns == pytest.approx(0.07))
    assert np.all(mc.variances == pytest.approx(0.0225))


def test_mc_two_asset_simplex():
    mu, cov = two_asset()
    mc = simulate_portfolios(mu, cov, n_portfolios=1000, seed=5)
    assert np.all(mc.weights >= 0.0) and np.all(mc.weights <= 1.0)
    assert np.allclose(mc.weights[:, 0] + mc.weights[:, 1], 1.0,
                       atol=1e-12)
    # returns lie within the feasible segment [min(mu), max(mu)]
    assert np.all(mc.expected_returns >= 0.06 - 1e-12)
    assert np.all(mc.expected_returns <= 0.10 + 1e-12)


# ---------------------------------------------------------------------------
# 9. MC validation
# ---------------------------------------------------------------------------

def test_mc_validation():
    mu, cov = two_asset()
    with pytest.raises(ValueError):
        simulate_portfolios(mu, cov, n_portfolios=0)
    with pytest.raises(TypeError):
        simulate_portfolios(mu, cov, n_portfolios=10.5)
    with pytest.raises(ValueError):
        simulate_portfolios(mu, cov, seed=-1)
    with pytest.raises(TypeError):
        simulate_portfolios(mu, cov, seed="42")
    with pytest.raises(TypeError):
        simulate_portfolios(mu, cov, seed=True)
    with pytest.raises(ValueError):
        simulate_portfolios(np.array([0.1, float("nan")]), cov)
    with pytest.raises(ValueError):
        simulate_portfolios(np.array([0.1, 0.06, 0.05]), cov)
    with pytest.raises(ValueError):
        simulate_portfolios(mu, np.array([[0.04, 0.004, 0.0],
                                          [0.004, 0.01, 0.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        simulate_portfolios(mu, np.array([[0.04, 0.005], [0.004, 0.01]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        simulate_portfolios(mu, np.array([[0.04, 0.05], [0.05, 0.01]]))


# ---------------------------------------------------------------------------
# 11A-D. Frontier feasibility + target matching + variance consistency
# ---------------------------------------------------------------------------

def test_frontier_constraints_and_targets():
    mu, cov = three_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=15, risk_free_rate=0.0)
    assert isinstance(fr, FrontierResult)
    assert fr.n_points == 15 and not fr.is_partial
    assert np.all(fr.success)
    assert np.all(fr.weights >= -popt.CONSTRAINT_TOL)
    assert np.allclose(fr.weights.sum(axis=1), 1.0,
                       atol=popt.CONSTRAINT_TOL)
    assert np.allclose(fr.expected_returns, fr.target_returns,
                       atol=1e-5)
    assert np.all(np.isfinite(fr.volatilities)) and np.all(fr.volatilities > 0)


def test_frontier_weights_reproduce_variance():
    mu, cov = two_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=7)
    assert not fr.is_partial
    for i in range(fr.n_points):
        assert fr.variances[i] == pytest.approx(
            pm.portfolio_variance(fr.weights[i], cov), rel=1e-6)
        assert fr.volatilities[i] == pytest.approx(
            math.sqrt(fr.variances[i]), rel=1e-6)


def test_frontier_target_range_is_feasible_interval():
    mu, cov = two_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=9)
    assert fr.target_returns[0] == pytest.approx(0.06)
    assert fr.target_returns[-1] == pytest.approx(0.10)
    assert np.all(np.diff(fr.target_returns) > 0)


# ---------------------------------------------------------------------------
# 11E/F. Determinism + two-asset analytical behavior
# ---------------------------------------------------------------------------

def test_frontier_determinism():
    mu, cov = three_asset()
    a = efficient_frontier(mu, cov, n_frontier_points=10)
    b = efficient_frontier(mu, cov, n_frontier_points=10)
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.target_returns, b.target_returns)


def test_frontier_two_asset_matches_unique_feasible():
    # two assets, distinct returns -> each target fixes weights uniquely:
    # wA = (R - 0.06) / 0.04 ; endpoints are the corner portfolios.
    mu, cov = two_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=5)
    assert not fr.is_partial
    assert list(fr.weights[0]) == pytest.approx([0.0, 1.0], abs=1e-4)
    assert list(fr.weights[-1]) == pytest.approx([1.0, 0.0], abs=1e-4)
    for i, target in enumerate(fr.target_returns):
        wA = (target - 0.06) / 0.04
        assert list(fr.weights[i]) == pytest.approx([wA, 1 - wA], abs=1e-4)


def test_frontier_minimum_vol_matches_task4b_minvar():
    # the unconstrained min-variance return always lies in [min(mu),
    # max(mu)]; the frontier's best point must be near it, and no
    # frontier point can beat the unconstrained optimum.
    mu, cov = three_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=25)
    assert not fr.is_partial
    mv = popt.minimum_variance_portfolio(cov, expected_returns=mu)
    assert np.min(fr.volatilities) >= mv.volatility - 1e-8
    idx = int(np.argmin(np.abs(fr.target_returns - mv.expected_return)))
    assert fr.volatilities[idx] <= mv.volatility + 1e-3


# ---------------------------------------------------------------------------
# 11G. Frontier risk no worse than a feasible portfolio at same target
# ---------------------------------------------------------------------------

def test_frontier_no_worse_than_feasible_blend():
    # feasible 50/50 blend of first two 3-asset names returns R=0.10;
    # the frontier point at R=0.10 must do at least as well (within tol).
    mu, cov = three_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=15)
    assert not fr.is_partial
    blend = np.array([0.5, 0.5, 0.0])  # return exactly 0.10
    assert float(blend @ mu) == pytest.approx(0.10)
    idx = int(np.argmin(np.abs(fr.target_returns - 0.10)))
    assert abs(fr.target_returns[idx] - 0.10) < 1e-9  # 0.10 is on-grid
    assert fr.volatilities[idx] <= pm.portfolio_volatility(blend, cov) + 1e-6


# ---------------------------------------------------------------------------
# Frontier degenerate + failure representation
# ---------------------------------------------------------------------------

def test_frontier_flat_mu_replicates_minvar():
    mu = np.array([0.07, 0.07])
    cov = np.array([[0.04, 0.004], [0.004, 0.01]])
    fr = efficient_frontier(mu, cov, n_frontier_points=6)
    assert not fr.is_partial
    assert np.all(fr.target_returns == pytest.approx(0.07))
    mv = popt.minimum_variance_portfolio(cov)
    for i in range(6):
        assert np.allclose(fr.weights[i], mv.weights, atol=1e-6)


def test_frontier_single_asset():
    fr = efficient_frontier(np.array([0.07]), np.array([[0.0225]]),
                            n_frontier_points=4)
    assert not fr.is_partial
    assert np.all(fr.weights == 1.0)
    assert np.all(fr.volatilities == pytest.approx(0.15))


def test_frontier_partial_status_when_solver_fails():
    mu, cov = two_asset()
    fr = efficient_frontier(mu, cov, n_frontier_points=5,
                            solver_options={"maxiter": 0})
    assert fr.is_partial
    assert fr.n_successful == 0
    assert not np.any(fr.success)
    assert np.all(np.isnan(fr.weights))


def test_frontier_validation():
    mu, cov = two_asset()
    with pytest.raises(ValueError):
        efficient_frontier(mu, cov, n_frontier_points=1)
    with pytest.raises(TypeError):
        efficient_frontier(mu, cov, solver_options=[1])
    with pytest.raises(ValueError):
        efficient_frontier(np.array([]), np.zeros((0, 0)))


# ---------------------------------------------------------------------------
# 12. Key-portfolio consistency with Task 4B
# ---------------------------------------------------------------------------

def test_key_portfolios_match_task4b():
    mu, cov = three_asset()
    rf = 0.02
    keys = key_portfolios(mu, cov, risk_free_rate=rf,
                          asset_names=["A", "B", "C"])
    assert set(keys) == {"equal_weight", "minimum_variance",
                         "maximum_sharpe"}
    assert keys["equal_weight"].weights_dict() == {"A": pytest.approx(1 / 3),
                                                   "B": pytest.approx(1 / 3),
                                                   "C": pytest.approx(1 / 3)}
    mv = popt.minimum_variance_portfolio(cov, expected_returns=mu,
                                         risk_free_rate=rf)
    ms = popt.maximum_sharpe_portfolio(mu, cov, risk_free_rate=rf)
    assert np.allclose(keys["minimum_variance"].weights, mv.weights,
                       atol=1e-10)
    assert np.allclose(keys["maximum_sharpe"].weights, ms.weights,
                       atol=1e-10)
    assert keys["minimum_variance"].variance == pytest.approx(mv.variance)
    assert keys["maximum_sharpe"].sharpe_ratio == pytest.approx(ms.sharpe_ratio)


def test_key_portfolios_consistent_with_mc_geometry():
    # optimizer solutions must be at least as good as any random sample.
    mu, cov = three_asset()
    rf = 0.02
    mc = simulate_portfolios(mu, cov, risk_free_rate=rf,
                             n_portfolios=2000, seed=0)
    keys = key_portfolios(mu, cov, risk_free_rate=rf)
    assert keys["minimum_variance"].variance <= np.min(mc.variances) + 1e-10
    assert keys["maximum_sharpe"].sharpe_ratio >= \
        np.nanmax(mc.sharpe_ratios) - 1e-6


# ---------------------------------------------------------------------------
# 13. Performance (vectorized default run is fast)
# ---------------------------------------------------------------------------

def test_default_mc_run_is_fast():
    mu, cov = three_asset()
    t0 = time.perf_counter()
    mc = simulate_portfolios(mu, cov)  # 20_000 default
    dt = time.perf_counter() - t0
    assert mc.weights.shape == (20_000, 3)
    assert dt < 20.0, f"MC 20k took {dt:.1f}s; vectorization broken?"


def test_default_frontier_run_is_fast():
    mu, cov = three_asset()
    t0 = time.perf_counter()
    fr = efficient_frontier(mu, cov)  # 50 points default
    dt = time.perf_counter() - t0
    assert fr.n_points == 50 and not fr.is_partial
    assert dt < 60.0, f"frontier 50 took {dt:.1f}s"
