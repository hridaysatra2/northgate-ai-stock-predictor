"""Tests for src.portfolio_math (Task 4A).

All hand-calculated expectations are derived inline in comments.
Floating-point comparisons use pytest.approx.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import portfolio_math as pm


# ---------------------------------------------------------------------------
# 1. Return statistics: sample-estimator convention, hand calculations
# ---------------------------------------------------------------------------

def test_mean_return_hand_calculated():
    # (0.01 + 0.02 + 0.03) / 3 = 0.02
    assert pm.mean_return([0.01, 0.02, 0.03]) == pytest.approx(0.02)


def test_return_variance_uses_sample_ddof1():
    # deviations from 0.02: -0.01, 0, +0.01 -> squares 1e-4, 0, 1e-4
    # sample var = 2e-4 / (3 - 1) = 1e-4 (population would give 6.67e-5)
    assert pm.return_variance([0.01, 0.02, 0.03]) == pytest.approx(1e-4)


def test_return_volatility_hand_calculated():
    assert pm.return_volatility([0.01, 0.02, 0.03]) == pytest.approx(0.01)
    assert pm.return_std([0.01, 0.02, 0.03]) == pytest.approx(0.01)


def test_covariance_hand_calculated():
    # a devs: -0.01, 0, 0.01 ; b devs: 0, -0.01, 0.01
    # products: 0, 0, 1e-4 -> sum 1e-4 / (3-1) = 5e-5
    assert pm.covariance([0.01, 0.02, 0.03],
                         [0.02, 0.01, 0.03]) == pytest.approx(5e-5)


def test_correlation_hand_calculated():
    # cov 5e-5 / (0.01 * 0.01) = 0.5
    assert pm.correlation([0.01, 0.02, 0.03],
                          [0.02, 0.01, 0.03]) == pytest.approx(0.5)


def test_example_c_covariance_from_vols_and_correlation():
    # sigma_A=0.20, sigma_B=0.30, rho=-0.5 -> cov = -0.5*0.2*0.3 = -0.03
    # Build series with exactly these moments via standardized pair.
    rng = np.random.default_rng(42)
    n = 5000
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    a = 0.20 * z1
    b = 0.30 * (-0.5 * z1 + math.sqrt(1 - 0.25) * z2)
    assert pm.covariance(a, b) == pytest.approx(-0.03, abs=2e-3)
    assert pm.correlation(a, b) == pytest.approx(-0.5, abs=0.03)


def test_covariance_matrix_matches_pairwise():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.015],
                       "B": [0.02, 0.01, 0.03, 0.025]})
    mat = pm.covariance_matrix(df)
    assert mat.shape == (2, 2)
    assert mat[0, 1] == pytest.approx(pm.covariance(df["A"], df["B"]))
    assert mat[0, 0] == pytest.approx(pm.return_variance(df["A"]))
    assert mat[1, 1] == pytest.approx(pm.return_variance(df["B"]))


def test_correlation_matrix_matches_pairwise():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.015],
                       "B": [0.02, 0.01, 0.03, 0.025]})
    mat = pm.correlation_matrix(df)
    assert mat.shape == (2, 2)
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[0, 1] == pytest.approx(pm.correlation(df["A"], df["B"]))


# ---------------------------------------------------------------------------
# 2. Expected return -- required example A
# ---------------------------------------------------------------------------

def test_example_a_expected_return():
    # 0.7*0.12 + 0.3*0.08 = 0.084 + 0.024 = 0.108
    assert pm.portfolio_expected_return([0.7, 0.3],
                                        [0.12, 0.08]) == pytest.approx(0.108)


def test_expected_return_does_not_normalize_weights():
    # weights summing to 2 must NOT be rescaled: 2 * 0.10 = 0.20
    assert pm.portfolio_expected_return([1.0, 1.0],
                                        [0.10, 0.10]) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 3-4. Variance / volatility -- required example B
# ---------------------------------------------------------------------------

def _example_b_cov():
    # var_A = 0.04, var_B = 0.01, cov = 0.2*0.2*0.1 = 0.004
    return np.array([[0.04, 0.004], [0.004, 0.01]])


def test_example_b_portfolio_variance():
    # 0.36*0.04 + 0.16*0.01 + 2*0.6*0.4*0.004
    # = 0.0144 + 0.0016 + 0.00192 = 0.01792
    assert pm.portfolio_variance([0.6, 0.4],
                                 _example_b_cov()) == pytest.approx(0.01792)


def test_example_b_portfolio_volatility():
    # sqrt(0.01792) approx 0.1338656
    assert pm.portfolio_volatility([0.6, 0.4],
                                   _example_b_cov()) == pytest.approx(0.1338656)
    assert pm.portfolio_volatility([0.6, 0.4], _example_b_cov()) == \
        pytest.approx(math.sqrt(0.01792))


def test_portfolio_variance_returns_scalar():
    out = pm.portfolio_variance([0.6, 0.4], _example_b_cov())
    assert isinstance(out, float)


def test_portfolio_volatility_tolerates_tiny_negative_artifact():
    # Near-singular matrix: w=[1,-1] gives var = 2 - 2*(1+5e-14) = -1e-13,
    # a tiny negative float artifact within NEG_VAR_TOL -> treated as 0.0.
    cov = np.array([[1.0, 1.0 + 5e-14], [1.0 + 5e-14, 1.0]])
    assert pm.portfolio_volatility([1.0, -1.0], cov) == pytest.approx(0.0)
    assert pm.portfolio_variance([1.0, -1.0], cov) == pytest.approx(0.0)


def test_portfolio_volatility_rejects_materially_negative_variance():
    bad = np.array([[-0.01, 0.0], [0.0, 0.01]])
    with pytest.raises(ValueError):
        pm.portfolio_volatility([1.0, 0.0], bad)
    with pytest.raises(ValueError):
        pm.portfolio_variance([1.0, 0.0], bad)


# ---------------------------------------------------------------------------
# 5. Sharpe -- required example D
# ---------------------------------------------------------------------------

def test_example_d_sharpe():
    # (0.15 - 0.05) / 0.20 = 0.5
    assert pm.sharpe_ratio(0.15, 0.05, 0.20) == pytest.approx(0.5)


def test_sharpe_negative_excess():
    assert pm.sharpe_ratio(0.02, 0.05, 0.10) == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# 6-7. Downside deviation / Sortino -- required example E
# ---------------------------------------------------------------------------

def test_example_e_downside_deviation_divisor_n():
    # returns [0.02,-0.01,0.03,-0.02,0.01], target 0:
    # shortfalls [0,-0.01,0,-0.02,0] -> squares sum 5e-4 / N=5 = 1e-4
    # DD = sqrt(1e-4) = 0.01  (divisor N, NOT n_downside=2 which'd give
    # sqrt(2.5e-4) = 0.01581 -- this test pins the N convention)
    r = [0.02, -0.01, 0.03, -0.02, 0.01]
    assert pm.downside_deviation(r, target=0.0) == pytest.approx(0.01)


def test_example_e_sortino():
    # mean = 0.006 ; (0.006 - 0) / 0.01 = 0.6
    r = [0.02, -0.01, 0.03, -0.02, 0.01]
    assert pm.sortino_ratio(r, target_return=0.0) == pytest.approx(0.6)


def test_sortino_nonzero_target_hand_calculated():
    # target 0.01: shortfalls [0,-0.02,0,-0.03,0] -> sum sq 13e-4
    # DD = sqrt(13e-4/5) = sqrt(2.6e-4); excess = 0.006-0.01 = -0.004
    r = [0.02, -0.01, 0.03, -0.02, 0.01]
    dd = math.sqrt(0.00026)
    assert pm.downside_deviation(r, target=0.01) == pytest.approx(dd)
    assert pm.sortino_ratio(r, target_return=0.01) == pytest.approx(-0.004 / dd)


def test_downside_deviation_all_above_target_is_zero():
    assert pm.downside_deviation([0.05, 0.06, 0.07],
                                 target=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Maximum drawdown -- required example F
# ---------------------------------------------------------------------------

def test_example_f_maximum_drawdown():
    # peaks 100,120,120,120,120 -> trough (90-120)/120 = -0.25
    assert pm.maximum_drawdown([100, 120, 110, 90, 105]) == pytest.approx(-0.25)


def test_maximum_drawdown_monotone_gain_is_zero():
    assert pm.maximum_drawdown([100, 110, 120]) == pytest.approx(0.0)


def test_maximum_drawdown_single_value_is_zero():
    assert pm.maximum_drawdown([100.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. Historical VaR -- required example G
# ---------------------------------------------------------------------------

def test_example_g_historical_var_return_units():
    # sorted 10 pts; alpha=0.10 -> idx (10-1)*0.1=0.9 between -0.05,-0.04
    # q = -0.05 + 0.9*0.01 = -0.041 ; VaR (positive-loss) = 0.041
    r = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    assert pm.historical_var(r, confidence=0.90) == pytest.approx(0.041)


def test_historical_var_scales_with_portfolio_value():
    r = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    assert pm.historical_var(r, confidence=0.90,
                             portfolio_value=1000.0) == pytest.approx(41.0)


def test_historical_var_positive_loss_convention():
    # deep-loss tail -> positive VaR (a loss amount, not a negative return)
    r = [-0.10, -0.09, -0.08, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    assert pm.historical_var(r, confidence=0.90) > 0.0


# ---------------------------------------------------------------------------
# 10. Annualization -- required example H
# ---------------------------------------------------------------------------

def test_example_h_annualized_volatility():
    # 0.01 * sqrt(252)
    assert pm.annualized_volatility(0.01) == pytest.approx(0.01 * math.sqrt(252))
    assert pm.annualized_volatility(0.01) == pytest.approx(0.15874507866387544)


def test_annualized_mean():
    assert pm.annualized_mean(0.001) == pytest.approx(0.252)
    assert pm.annualized_mean(0.02, periods_per_year=12) == pytest.approx(0.24)


# ---------------------------------------------------------------------------
# 11. Validation / error cases
# ---------------------------------------------------------------------------

def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        pm.mean_return([])
    with pytest.raises(ValueError):
        pm.return_variance([])
    with pytest.raises(ValueError):
        pm.downside_deviation([], target=0.0)
    with pytest.raises(ValueError):
        pm.maximum_drawdown([])
    with pytest.raises(ValueError):
        pm.historical_var([], confidence=0.95)
    with pytest.raises(ValueError):
        pm.portfolio_expected_return([], [])


def test_nan_never_imputed():
    with pytest.raises(ValueError):
        pm.mean_return([0.01, float("nan"), 0.03])
    with pytest.raises(ValueError):
        pm.return_variance([0.01, float("nan")])
    with pytest.raises(ValueError):
        pm.covariance([0.01, float("nan")], [0.01, 0.02])
    with pytest.raises(ValueError):
        pm.downside_deviation([0.01, float("nan")], target=0.0)
    with pytest.raises(ValueError):
        pm.maximum_drawdown([100.0, float("nan")])
    with pytest.raises(ValueError):
        pm.historical_var([0.01, float("nan")], confidence=0.95)
    with pytest.raises(ValueError):
        pm.portfolio_expected_return([0.5, float("nan")], [0.1, 0.1])
    with pytest.raises(ValueError):
        pm.portfolio_variance([0.5, 0.5], np.array([[0.01, np.nan],
                                                    [np.nan, 0.01]]))


def test_inf_rejected():
    with pytest.raises(ValueError):
        pm.mean_return([0.01, float("inf")])
    with pytest.raises(ValueError):
        pm.portfolio_expected_return([0.5, float("inf")], [0.1, 0.1])
    with pytest.raises(ValueError):
        pm.sharpe_ratio(0.1, 0.02, float("inf"))


def test_insufficient_observations_raise():
    with pytest.raises(ValueError):
        pm.return_variance([0.05])
    with pytest.raises(ValueError):
        pm.return_volatility([0.05])
    with pytest.raises(ValueError):
        pm.covariance([0.01], [0.02])
    with pytest.raises(ValueError):
        pm.correlation([0.01, 0.02], [0.03, 0.04][:1])
    with pytest.raises(ValueError):
        pm.downside_deviation([0.05], target=0.0)
    with pytest.raises(ValueError):
        pm.sortino_ratio([0.05], target_return=0.0)
    with pytest.raises(ValueError):
        pm.historical_var([0.05], confidence=0.95)


def test_mismatched_dimensions_raise():
    with pytest.raises(ValueError):
        pm.portfolio_expected_return([0.5, 0.5], [0.1])
    with pytest.raises(ValueError):
        pm.portfolio_variance([0.5, 0.5, 0.0], _example_b_cov())
    with pytest.raises(ValueError):
        pm.portfolio_volatility([0.5], _example_b_cov())
    with pytest.raises(ValueError):
        pm.covariance([0.01, 0.02, 0.03], [0.01, 0.02])


def test_non_square_covariance_raises():
    with pytest.raises(ValueError):
        pm.portfolio_variance([0.5, 0.5], np.array([[0.04, 0.004, 0.0],
                                                    [0.004, 0.01, 0.0]]))
    with pytest.raises(ValueError):
        pm.portfolio_volatility([0.5, 0.5], np.array([0.04, 0.01]))


def test_invalid_confidence_raises():
    r = [0.01, -0.02, 0.03, -0.01, 0.02]
    for bad in (0.0, 1.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError):
            pm.historical_var(r, confidence=bad)


def test_invalid_portfolio_value_raises():
    r = [0.01, -0.02, 0.03, -0.01, 0.02]
    for bad in (0.0, -100.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            pm.historical_var(r, confidence=0.95, portfolio_value=bad)


def test_zero_volatility_sharpe_raises():
    with pytest.raises(ValueError):
        pm.sharpe_ratio(0.10, 0.02, 0.0)
    with pytest.raises(ValueError):
        pm.sharpe_ratio(0.10, 0.02, -0.01)


def test_zero_downside_deviation_sortino_raises():
    with pytest.raises(ValueError):
        pm.sortino_ratio([0.05, 0.06, 0.07], target_return=0.0)


def test_zero_variance_correlation_raises():
    with pytest.raises(ValueError):
        pm.correlation([0.05, 0.05, 0.05], [0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        pm.correlation_matrix(np.array([[0.05, 0.01],
                                        [0.05, 0.02],
                                        [0.05, 0.03]]))


def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        pm.portfolio_variance([], _example_b_cov())
    with pytest.raises(ValueError):
        pm.portfolio_variance([0.5, float("inf")], _example_b_cov())


def test_invalid_drawdown_values_raise():
    with pytest.raises(ValueError):
        pm.maximum_drawdown([100.0, 0.0, 90.0])
    with pytest.raises(ValueError):
        pm.maximum_drawdown([100.0, -5.0])
    with pytest.raises(ValueError):
        pm.maximum_drawdown([100.0, float("inf")])


def test_non_numeric_inputs_raise_type_error():
    with pytest.raises(TypeError):
        pm.mean_return(["a", "b"])
    with pytest.raises(TypeError):
        pm.sharpe_ratio("x", 0.02, 0.1)
    with pytest.raises(TypeError):
        pm.sortino_ratio([0.01, 0.02], target_return="tgt")
    with pytest.raises(TypeError):
        pm.historical_var([0.01, 0.02], confidence="high")


def test_negative_annualized_volatility_raises():
    with pytest.raises(ValueError):
        pm.annualized_volatility(-0.01)


def test_pandas_inputs_accepted():
    s = pd.Series([0.01, 0.02, 0.03])
    assert pm.mean_return(s) == pytest.approx(0.02)
    assert pm.return_variance(s) == pytest.approx(1e-4)
    assert pm.portfolio_expected_return(pd.Series([0.7, 0.3]),
                                        pd.Series([0.12, 0.08])) == \
        pytest.approx(0.108)
