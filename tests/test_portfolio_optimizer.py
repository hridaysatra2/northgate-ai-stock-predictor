"""Tests for src.portfolio_optimizer (Task 4B).

Deterministic, numerical-input only. No market data, no dates, no
backtesting. Floating-point comparisons use pytest.approx / allclose,
never exact equality on optimizer output.
"""

import math

import numpy as np
import pytest

from src import portfolio_math as pm
from src.portfolio_optimizer import (
    CONSTRAINT_TOL,
    OptimizationError,
    OptimizationResult,
    equal_weight_portfolio,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
)


def two_asset_cov():
    """vol A=20%, vol B=10%, corr=0.2 -> cov=0.004."""
    return np.array([[0.04, 0.004], [0.004, 0.01]])


def assert_long_only_fully_invested(weights):
    w = np.asarray(weights, dtype=float)
    assert np.all(w >= -CONSTRAINT_TOL), f"short weights: {w}"
    assert float(np.sum(w)) == pytest.approx(1.0, abs=CONSTRAINT_TOL)


# 7. Equal weight

def test_equal_weight_values():
    r = equal_weight_portfolio(4)
    assert len(r.weights) == 4
    assert list(r.weights) == pytest.approx([0.25] * 4)
    assert_long_only_fully_invested(r.weights)
    assert r.objective_name == "equal_weight"
    assert r.success is True
    assert r.solver_status == 0
    assert "no optimizer" in r.solver_message
    assert r.constraint_residual == pytest.approx(0.0, abs=1e-12)


def test_equal_weight_metrics_match_portfolio_math():
    cov, mu = two_asset_cov(), np.array([0.10, 0.06])
    r = equal_weight_portfolio(2, covariance_matrix=cov,
                               expected_returns=mu, risk_free_rate=0.02)
    # hand: var = .25*.04+.25*.01+2*.25*.004 = .0145 ; ret = .08
    assert r.variance == pytest.approx(0.0145)
    assert r.volatility == pytest.approx(math.sqrt(0.0145))
    assert r.expected_return == pytest.approx(0.08)
    assert r.sharpe_ratio == pytest.approx((0.08 - 0.02) / math.sqrt(0.0145))
    # cross-check against Task 4A directly
    assert r.variance == pytest.approx(pm.portfolio_variance([0.5, 0.5], cov))


def test_equal_weight_no_matrix_gives_nan_metrics():
    r = equal_weight_portfolio(3)
    assert_long_only_fully_invested(r.weights)
    assert math.isnan(r.variance) and math.isnan(r.volatility)
    assert math.isnan(r.expected_return) and math.isnan(r.sharpe_ratio)


def test_equal_weight_names_and_weights_dict():
    r = equal_weight_portfolio(2, asset_names=["AAA", "BBB"])
    assert r.asset_names == ("AAA", "BBB")
    assert r.weights_dict() == {"AAA": pytest.approx(0.5),
                                "BBB": pytest.approx(0.5)}
    # ordering preserved
    assert list(r.weights_dict()) == ["AAA", "BBB"]


def test_equal_weight_invalid_n():
    with pytest.raises(ValueError):
        equal_weight_portfolio(0)
    with pytest.raises(ValueError):
        equal_weight_portfolio(-2)
    with pytest.raises(TypeError):
        equal_weight_portfolio(2.5)
    with pytest.raises(TypeError):
        equal_weight_portfolio(True)


# 8. Minimum variance

def test_minvar_two_asset_matches_analytical():
    # interior solution: wA = (vB-c)/(vA+vB-2c) = .006/.042 = 1/7
    r = minimum_variance_portfolio(two_asset_cov())
    assert r.success is True
    assert_long_only_fully_invested(r.weights)
    assert list(r.weights) == pytest.approx([1 / 7, 6 / 7], abs=1e-4)
    # hand: (0.04 + 0.048 + 0.36)/49 = 0.448/49
    assert r.variance == pytest.approx(0.448 / 49, rel=1e-4)
    assert math.isfinite(r.volatility)


def test_minvar_beats_equal_weight_variance():
    cov = two_asset_cov()
    r = minimum_variance_portfolio(cov)
    eq = equal_weight_portfolio(2, covariance_matrix=cov)
    assert r.variance <= eq.variance + 1e-10


def test_minvar_identical_assets_non_unique_ok():
    # rank-1 PSD: every feasible w has var 0.04; accept any valid solution.
    cov = np.array([[0.04, 0.04], [0.04, 0.04]])
    mu = np.array([0.08, 0.08])
    r = minimum_variance_portfolio(cov, expected_returns=mu)
    assert r.success is True
    assert_long_only_fully_invested(r.weights)
    assert r.variance == pytest.approx(0.04, abs=1e-8)
    assert r.variance <= 0.04 + 1e-8  # no worse than any feasible point


def test_minvar_reports_expected_return_and_sharpe():
    r = minimum_variance_portfolio(two_asset_cov(),
                                   expected_returns=np.array([0.10, 0.06]),
                                   risk_free_rate=0.02)
    assert r.expected_return == pytest.approx(
        pm.portfolio_expected_return(r.weights, [0.10, 0.06]))
    assert r.sharpe_ratio == pytest.approx(
        pm.sharpe_ratio(r.expected_return, 0.02, r.volatility))


def test_minvar_determinism():
    cov = np.array([[0.04, 0.008, 0.002],
                    [0.008, 0.09, -0.01],
                    [0.002, -0.01, 0.0225]])
    a = minimum_variance_portfolio(cov)
    b = minimum_variance_portfolio(cov)
    assert np.allclose(a.weights, b.weights, atol=1e-10)


def test_minvar_result_structure():
    r = minimum_variance_portfolio(two_asset_cov(), asset_names=["A", "B"])
    assert isinstance(r, OptimizationResult)
    d = r.to_dict()
    for key in ("weights", "expected_return", "variance", "volatility",
                "sharpe_ratio", "objective_name", "success",
                "solver_status", "solver_message", "constraint_residual"):
        assert key in d
    assert d["objective_name"] == "minimum_variance"
    assert d["weights"] == pytest.approx(list(r.weights))
    assert r.n_assets == 2
    assert r.weights_dict() == {"A": pytest.approx(r.weights[0]),
                                "B": pytest.approx(r.weights[1])}


# 9. Maximum Sharpe

def sharpe_case():
    mu = np.array([0.12, 0.08, 0.05])
    cov = np.array([[0.04, 0.006, 0.002],
                    [0.006, 0.0225, 0.001],
                    [0.002, 0.001, 0.01]])
    return mu, cov


def test_maxsharpe_valid_and_beats_equal_weight():
    mu, cov = sharpe_case()
    rf = 0.02
    r = maximum_sharpe_portfolio(mu, cov, risk_free_rate=rf)
    assert r.success is True
    assert_long_only_fully_invested(r.weights)
    assert math.isfinite(r.sharpe_ratio)
    eq = equal_weight_portfolio(3, covariance_matrix=cov,
                               expected_returns=mu, risk_free_rate=rf)
    assert r.sharpe_ratio >= eq.sharpe_ratio - 1e-6


def test_maxsharpe_metrics_internally_consistent():
    mu, cov = sharpe_case()
    r = maximum_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
    assert r.expected_return == pytest.approx(
        pm.portfolio_expected_return(r.weights, mu))
    assert r.variance == pytest.approx(pm.portfolio_variance(r.weights, cov))
    assert r.sharpe_ratio == pytest.approx(
        pm.sharpe_ratio(r.expected_return, 0.02, r.volatility))


def test_maxsharpe_determinism():
    mu, cov = sharpe_case()
    a = maximum_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
    b = maximum_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
    assert np.allclose(a.weights, b.weights, atol=1e-10)


def test_maxsharpe_prefers_high_return_low_vol():
    # near-riskless high-return asset dominates: weight concentrates on A.
    mu = np.array([0.15, 0.04])
    cov = np.array([[0.0025, 0.0], [0.0, 0.04]])
    r = maximum_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
    assert r.success is True
    assert_long_only_fully_invested(r.weights)
    assert r.weights[0] > 0.9


# 10. Edge cases

def test_single_asset_minvar_returns_one():
    r = minimum_variance_portfolio(np.array([[0.04]]),
                                   expected_returns=np.array([0.08]))
    assert list(r.weights) == pytest.approx([1.0])
    assert r.variance == pytest.approx(0.04)
    assert r.success is True


def test_single_asset_maxsharpe_returns_one():
    r = maximum_sharpe_portfolio(np.array([0.08]), np.array([[0.04]]),
                                 risk_free_rate=0.02)
    assert list(r.weights) == pytest.approx([1.0])
    assert r.sharpe_ratio == pytest.approx(0.3)  # (0.08-0.02)/0.2
    assert r.success is True


def test_single_asset_zero_variance_maxsharpe_raises():
    with pytest.raises(OptimizationError):
        maximum_sharpe_portfolio(np.array([0.05]), np.array([[0.0]]),
                                 risk_free_rate=0.02)


def test_mismatched_dimensions_raise():
    with pytest.raises(ValueError):
        minimum_variance_portfolio(np.eye(2),
                                   expected_returns=np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(np.array([0.1, 0.2, 0.3]), np.eye(2))


def test_nan_inf_inputs_raise():
    bad_mu = np.array([0.1, float("nan")])
    bad_cov = np.array([[0.04, float("nan")], [float("nan"), 0.01]])
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(bad_mu, two_asset_cov(), risk_free_rate=0.02)
    with pytest.raises(ValueError):
        minimum_variance_portfolio(bad_cov)
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]), bad_cov,
                                 risk_free_rate=0.02)
    with pytest.raises(ValueError):
        minimum_variance_portfolio(np.array([[float("inf"), 0.0],
                                             [0.0, 0.01]]))
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]), two_asset_cov(),
                                 risk_free_rate=float("inf"))
    with pytest.raises(ValueError):
        minimum_variance_portfolio(two_asset_cov(), risk_free_rate=float("nan"))


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(np.array([]), np.zeros((0, 0)))
    with pytest.raises(ValueError):
        minimum_variance_portfolio(np.zeros((0, 0)))
    with pytest.raises(TypeError):
        maximum_sharpe_portfolio("not-a-vector", two_asset_cov())


def test_non_square_covariance_raises():
    with pytest.raises(ValueError):
        minimum_variance_portfolio(np.array([[0.04, 0.004, 0.0],
                                             [0.004, 0.01, 0.0]]))
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]),
                                 np.array([0.04, 0.01]))


def test_asymmetric_covariance_raises_not_silently_fixed():
    asym = np.array([[0.04, 0.005], [0.004, 0.01]])
    with pytest.raises(ValueError, match="symmetric"):
        minimum_variance_portfolio(asym)
    with pytest.raises(ValueError, match="symmetric"):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]), asym,
                                 risk_free_rate=0.0)


def test_indefinite_covariance_raises():
    # det < 0 -> min eigenvalue materially negative; equal-weight var is
    # still positive (0.0375), so only the eig check catches this.
    bad = np.array([[0.04, 0.05], [0.05, 0.01]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        minimum_variance_portfolio(bad)
    with pytest.raises(ValueError, match="positive semidefinite"):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]), bad,
                                 risk_free_rate=0.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        equal_weight_portfolio(2, covariance_matrix=bad)


def test_near_singular_covariance_still_optimizes():
    cov = np.array([[0.04, 0.0396], [0.0396, 0.04]])  # corr 0.99
    r = minimum_variance_portfolio(cov)
    assert r.success is True
    assert_long_only_fully_invested(r.weights)
    assert math.isfinite(r.variance)
    s = maximum_sharpe_portfolio(np.array([0.10, 0.09]), cov,
                                 risk_free_rate=0.02)
    assert s.success is True
    assert_long_only_fully_invested(s.weights)


def test_optimizer_failure_reported_not_hidden():
    # maxiter=0 forces SLSQP 'Iteration limit reached' (status 9).
    r = minimum_variance_portfolio(two_asset_cov(),
                                   solver_options={"maxiter": 0})
    assert r.success is False
    assert r.solver_status == 9
    assert r.objective_name == "minimum_variance"
    s = maximum_sharpe_portfolio(np.array([0.1, 0.06]), two_asset_cov(),
                                 risk_free_rate=0.02,
                                 solver_options={"maxiter": 0})
    assert s.success is False


def test_degenerate_zero_covariance_maxsharpe_raises():
    with pytest.raises(ValueError, match="ill-posed"):
        maximum_sharpe_portfolio(np.array([0.05, 0.05]),
                                 np.zeros((2, 2)), risk_free_rate=0.02)


def test_invalid_x0_raises():
    cov = two_asset_cov()
    with pytest.raises(ValueError):
        minimum_variance_portfolio(cov, x0=np.array([0.5, 0.5, 0.0]))
    with pytest.raises(ValueError):
        minimum_variance_portfolio(cov, x0=np.array([1.2, -0.2]))
    with pytest.raises(ValueError):
        minimum_variance_portfolio(cov, x0=np.array([0.5, 0.4]))
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]), cov,
                                 risk_free_rate=0.0,
                                 x0=np.array([0.5, float("nan")]))


def test_valid_custom_x0_accepted():
    cov = two_asset_cov()
    a = minimum_variance_portfolio(cov, x0=np.array([0.9, 0.1]))
    b = minimum_variance_portfolio(cov)
    assert a.success and np.allclose(a.weights, b.weights, atol=1e-6)


def test_invalid_names_and_options_raise():
    with pytest.raises(ValueError):
        minimum_variance_portfolio(two_asset_cov(), asset_names=["only-one"])
    with pytest.raises(TypeError):
        minimum_variance_portfolio(two_asset_cov(), asset_names=["A", 5])
    with pytest.raises(TypeError):
        minimum_variance_portfolio(two_asset_cov(), solver_options=[1, 2])
    r = minimum_variance_portfolio(two_asset_cov())  # no names attached
    with pytest.raises(ValueError):
        r.weights_dict()


def test_invalid_risk_free_type_raises():
    with pytest.raises(TypeError):
        maximum_sharpe_portfolio(np.array([0.1, 0.06]), two_asset_cov(),
                                 risk_free_rate="low")


def test_constraint_residual_reported():
    r = minimum_variance_portfolio(two_asset_cov())
    assert r.constraint_residual <= CONSTRAINT_TOL
    assert r.constraint_residual >= 0.0
