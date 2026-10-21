"""Portfolio and risk mathematics (Task 4A).

Pure-math foundation for Task 4B portfolio construction/backtesting.
No optimisation, no frontier, no backtesting, no I/O, no model training.

Conventions (applied consistently throughout):
  * Returns are simple per-period fractional returns (e.g. 0.01 = +1%).
    All inputs to any one function must share the same frequency/units.
  * Estimator: SAMPLE statistics (ddof=1, Bessel-corrected) for variance,
    volatility, covariance and correlation. The mean is the plain
    arithmetic mean. Rationale: historical windows are samples used to
    estimate forward-looking parameters (standard in finance).
  * Missing data: NaN is NEVER imputed or replaced with zero. Any NaN
    raises ValueError. Non-finite (Inf) inputs raise ValueError.
  * Minimum observations: mean needs >= 1; every dispersion/co-movement
    statistic (variance, volatility, covariance, correlation, downside
    deviation, Sortino, VaR) needs >= 2; single observations raise
    ValueError.
  * VaR: HISTORICAL quantile, returned as a POSITIVE loss amount in
    return units (VaR = -quantile(returns, 1 - confidence)); scaled by
    ``portfolio_value`` when given. A negative result means the quantile
    itself was a gain (no loss at that confidence).
  * Downside deviation: target (MAR) semideviation with divisor N over
    ALL observations: DD = sqrt(mean(min(0, r_i - target)^2)).
  * Max drawdown: negative fraction of running peak, e.g. -0.25 = -25%.
  * Annualization is NEVER implicit. Explicit helpers use 252 trading
    days for daily inputs: ann_mean = mean * 252, ann_vol = vol * sqrt(252).

Edge-case behaviour: empty input, NaN/Inf, mismatched dimensions,
non-square covariance, invalid confidence, zero volatility / zero
downside deviation all raise ValueError/TypeError with explicit
messages. Nothing is silently corrected or normalised.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "NEG_VAR_TOL",
    "mean_return",
    "return_variance",
    "return_volatility",
    "return_std",
    "covariance",
    "correlation",
    "covariance_matrix",
    "correlation_matrix",
    "portfolio_expected_return",
    "portfolio_variance",
    "portfolio_volatility",
    "sharpe_ratio",
    "downside_deviation",
    "sortino_ratio",
    "maximum_drawdown",
    "historical_var",
    "annualized_mean",
    "annualized_volatility",
]

TRADING_DAYS_PER_YEAR = 252
# Tolerance for tiny negative portfolio-variance float artifacts only.
NEG_VAR_TOL = 1e-12


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _as_float_1d(values, *, name: str) -> np.ndarray:
    """Coerce array-like/Series to a 1-D float ndarray.

    Raises:
        TypeError: if input cannot be interpreted as numeric 1-D data.
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.to_numpy()
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric 1-D data: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    return arr


def _reject_empty(arr: np.ndarray, *, name: str) -> None:
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _reject_non_finite(arr: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} must contain only finite values (no NaN/Inf); "
            "missing values are never imputed."
        )


def _require_n(arr: np.ndarray, minimum: int, *, name: str) -> None:
    if arr.size < minimum:
        raise ValueError(
            f"{name} needs at least {minimum} observations, got {arr.size}."
        )


def _clean_1d(values, *, name: str, minimum: int) -> np.ndarray:
    """Validate a 1-D return/value series and return float ndarray."""
    arr = _as_float_1d(values, name=name)
    _reject_empty(arr, name=name)
    _reject_non_finite(arr, name=name)  # NaN/Inf rejected, never filled
    _require_n(arr, minimum, name=name)
    return arr


# ---------------------------------------------------------------------------
# 1. Return statistics (sample estimators, ddof=1)
# ---------------------------------------------------------------------------

def _is_degenerate_scale(std: float, scale: float) -> bool:
    """True when a dispersion is zero up to float noise.

    Constant float inputs (e.g. three 0.05 values) can yield std ~1e-18
    instead of exactly 0.0, so an exact ``== 0.0`` check misses them.
    A scale-relative tolerance treats these as zero variance while
    leaving genuine financial dispersions untouched.
    """
    return bool(std <= 1e-9 * max(float(scale), 1e-300))

def mean_return(returns) -> float:
    """Arithmetic mean of per-period fractional returns.

    Formula: mean = sum(r_i) / N. Needs >= 1 observation.
    """
    arr = _clean_1d(returns, name="returns", minimum=1)
    return float(np.mean(arr))


def return_variance(returns) -> float:
    """SAMPLE variance (ddof=1) of returns.

    Formula: sum((r_i - mean)^2) / (N - 1). Needs >= 2 observations.
    """
    arr = _clean_1d(returns, name="returns", minimum=2)
    return float(np.var(arr, ddof=1))


def return_volatility(returns) -> float:
    """SAMPLE standard deviation (ddof=1) of returns.

    Formula: sqrt(sample variance). Needs >= 2 observations.
    """
    return float(math.sqrt(return_variance(returns)))


def return_std(returns) -> float:
    """Alias of :func:`return_volatility` (sample std, ddof=1)."""
    return return_volatility(returns)


def covariance(a, b) -> float:
    """SAMPLE covariance (ddof=1) between two return series.

    Formula: sum((a_i - mean_a)(b_i - mean_b)) / (N - 1).
    Both series must have equal length >= 2.
    """
    x = _clean_1d(a, name="a", minimum=2)
    y = _clean_1d(b, name="b", minimum=2)
    if x.shape != y.shape:
        raise ValueError(
            f"a and b must have the same length, got {x.shape} vs {y.shape}."
        )
    return float(np.cov(x, y, ddof=1)[0, 1])


def correlation(a, b) -> float:
    """SAMPLE (Pearson) correlation between two return series.

    Formula: cov(a, b) / (std_a * std_b). Raises ValueError when either
    series has zero variance (correlation undefined) instead of
    returning NaN.
    """
    x = _clean_1d(a, name="a", minimum=2)
    y = _clean_1d(b, name="b", minimum=2)
    if x.shape != y.shape:
        raise ValueError(
            f"a and b must have the same length, got {x.shape} vs {y.shape}."
        )
    std_x = float(np.std(x, ddof=1))
    std_y = float(np.std(y, ddof=1))
    if _is_degenerate_scale(std_x, np.max(np.abs(x))) or \
            _is_degenerate_scale(std_y, np.max(np.abs(y))):
        raise ValueError("correlation is undefined when a series has zero variance.")
    corr = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(corr):
        raise ValueError("correlation is non-finite (degenerate input).")
    return corr


def _as_2d_float(values, *, name: str) -> np.ndarray:
    """Coerce DataFrame/2-D array-like to a 2-D float ndarray."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.to_numpy()
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric 2-D data: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")
    if arr.shape[0] < 2:
        raise ValueError(
            f"{name} needs at least 2 observations (rows), got {arr.shape[0]}."
        )
    return arr


def covariance_matrix(returns) -> np.ndarray:
    """SAMPLE covariance matrix (ddof=1) of a returns panel.

    Input: (n_obs, n_assets) array/DataFrame, columns = assets.
    Returns (n_assets, n_assets) ndarray with cov[i, j] = sample
    covariance of columns i and j.
    """
    arr = _as_2d_float(returns, name="returns")
    return np.asarray(np.cov(arr, rowvar=False, ddof=1), dtype=float)


def correlation_matrix(returns) -> np.ndarray:
    """SAMPLE Pearson correlation matrix of a returns panel.

    Raises ValueError if any asset has zero variance (undefined) or if
    the result is non-finite.
    """
    arr = _as_2d_float(returns, name="returns")
    stds = np.std(arr, axis=0, ddof=1)
    scales = np.max(np.abs(arr), axis=0)
    if any(_is_degenerate_scale(s, c) for s, c in zip(stds, scales)):
        raise ValueError(
            "correlation matrix is undefined: an asset has zero variance."
        )
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    if not np.all(np.isfinite(corr)):
        raise ValueError("correlation matrix is non-finite (degenerate input).")
    return corr


# ---------------------------------------------------------------------------
# 2-4. Portfolio moments
# ---------------------------------------------------------------------------

def _clean_weights(weights, n: int | None = None) -> np.ndarray:
    w = _as_float_1d(weights, name="weights")
    _reject_empty(w, name="weights")
    _reject_non_finite(w, name="weights")
    if n is not None and w.size != n:
        raise ValueError(
            f"weights length ({w.size}) must match expected dimension ({n})."
        )
    return w


def _clean_cov_matrix(cov) -> np.ndarray:
    if isinstance(cov, pd.DataFrame):
        cov = cov.to_numpy()
    try:
        mat = np.asarray(cov, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"covariance_matrix must be numeric: {exc}") from exc
    if mat.ndim != 2:
        raise ValueError(
            f"covariance_matrix must be 2-D square, got shape {mat.shape}."
        )
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"covariance_matrix must be square, got shape {mat.shape}."
        )
    if mat.shape[0] == 0:
        raise ValueError("covariance_matrix must not be empty.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("covariance_matrix must contain only finite values.")
    return mat


def portfolio_expected_return(weights, expected_returns) -> float:
    """Portfolio expected return: E(Rp) = w^T mu.

    Units: same per-period fractional-return units as ``expected_returns``
    (e.g. monthly in -> monthly out). Weights are used AS GIVEN; this
    function never normalises them (constraints belong to the Task 4B
    optimiser).
    """
    w = _clean_weights(weights)
    mu = _as_float_1d(expected_returns, name="expected_returns")
    _reject_empty(mu, name="expected_returns")
    _reject_non_finite(mu, name="expected_returns")
    if w.size != mu.size:
        raise ValueError(
            f"weights ({w.size}) and expected_returns ({mu.size}) "
            "must have the same length."
        )
    return float(np.dot(w, mu))


def portfolio_variance(weights, covariance_matrix) -> float:
    """Portfolio variance: sigma_p^2 = w^T Sigma w. Returns a scalar float.

    The covariance matrix is used as given (never modified/symmetrised/
    regularised here). Tiny negative results within ``NEG_VAR_TOL`` of
    zero are clamped to 0.0 here so downstream sqrt stays well-defined;
    materially negative values raise ValueError (inconsistent inputs).
    """
    w = _clean_weights(weights)
    mat = _clean_cov_matrix(covariance_matrix)
    if w.size != mat.shape[0]:
        raise ValueError(
            f"weights length ({w.size}) must match covariance_matrix "
            f"dimension ({mat.shape[0]})."
        )
    var = float(w @ mat @ w)
    if not math.isfinite(var):
        raise ValueError("portfolio variance is non-finite; check inputs.")
    if var < 0.0 and var >= -NEG_VAR_TOL:
        return 0.0
    if var < 0.0:
        raise ValueError(
            f"portfolio variance is materially negative ({var}); "
            "covariance matrix / weights are inconsistent."
        )
    return var


def portfolio_volatility(weights, covariance_matrix) -> float:
    """Portfolio volatility: sigma_p = sqrt(w^T Sigma w).

    Only tiny negative float artifacts (within ``NEG_VAR_TOL``) are
    tolerated (treated as zero); materially negative variance raises
    ValueError rather than sqrt(abs(x)).
    """
    w = _clean_weights(weights)
    mat = _clean_cov_matrix(covariance_matrix)
    if w.size != mat.shape[0]:
        raise ValueError(
            f"weights length ({w.size}) must match covariance_matrix "
            f"dimension ({mat.shape[0]})."
        )
    var = float(w @ mat @ w)
    if not math.isfinite(var):
        raise ValueError("portfolio variance is non-finite; check inputs.")
    if var < 0.0 and var >= -NEG_VAR_TOL:
        return 0.0
    if var < 0.0:
        raise ValueError(
            f"portfolio variance is materially negative ({var}); "
            "refusing to take sqrt."
        )
    return float(math.sqrt(var))


# ---------------------------------------------------------------------------
# 5. Sharpe ratio
# ---------------------------------------------------------------------------

def sharpe_ratio(portfolio_return, risk_free_rate, portfolio_volatility) -> float:
    """Sharpe ratio: (Rp - Rf) / sigma_p.

    Frequency convention: ``portfolio_return`` and ``risk_free_rate``
    must share the same return frequency as ``portfolio_volatility``
    (e.g. all annualised or all monthly). No annualization is applied.
    Zero or negative volatility raises ValueError (never returns
    infinity/NaN silently). All inputs must be finite scalars.
    """
    for label, value in (("portfolio_return", portfolio_return),
                         ("risk_free_rate", risk_free_rate),
                         ("portfolio_volatility", portfolio_volatility)):
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be a numeric scalar: {exc}") from exc
        if not math.isfinite(v):
            raise ValueError(f"{label} must be finite, got {value}.")
    rp = float(portfolio_return)
    rf = float(risk_free_rate)
    vol = float(portfolio_volatility)
    if vol <= 0.0:
        raise ValueError(
            f"portfolio_volatility must be strictly positive, got {vol}."
        )
    return (rp - rf) / vol


# ---------------------------------------------------------------------------
# 6-7. Downside deviation and Sortino ratio
# ---------------------------------------------------------------------------

def downside_deviation(returns, target: float = 0.0) -> float:
    """Target downside deviation (MAR semideviation), divisor N.

    Convention (documented, not hidden): with target T (minimum
    acceptable return, same units/frequency as ``returns``),

        DD = sqrt( sum(min(0, r_i - T)^2) / N ),

    i.e. squared shortfalls averaged over ALL N observations (standard
    Sortino denominator), then square-rooted. Upside observations
    contribute 0. Needs >= 2 observations. NaN/Inf raise ValueError.
    """
    arr = _clean_1d(returns, name="returns", minimum=2)
    try:
        tgt = float(target)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"target must be a numeric scalar: {exc}") from exc
    if not math.isfinite(tgt):
        raise ValueError(f"target must be finite, got {target}.")
    shortfalls = np.minimum(arr - tgt, 0.0)
    return float(math.sqrt(float(np.mean(shortfalls ** 2))))


def sortino_ratio(returns, target_return: float = 0.0) -> float:
    """Sortino ratio: (mean(returns) - T) / downside_deviation.

    Uses the module's downside-deviation convention (divisor N over all
    observations; see :func:`downside_deviation`). ``target_return`` is
    the MAR in the same units/frequency as ``returns``. Zero downside
    deviation raises ValueError (never returns infinity/NaN silently).
    """
    arr = _clean_1d(returns, name="returns", minimum=2)
    try:
        tgt = float(target_return)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"target_return must be a numeric scalar: {exc}") from exc
    if not math.isfinite(tgt):
        raise ValueError(f"target_return must be finite, got {target_return}.")
    dd = downside_deviation(arr, target=tgt)
    if dd == 0.0:
        raise ValueError(
            "downside deviation is zero; Sortino ratio is undefined."
        )
    return (float(np.mean(arr)) - tgt) / dd


# ---------------------------------------------------------------------------
# 8. Maximum drawdown
# ---------------------------------------------------------------------------

def maximum_drawdown(values) -> float:
    """Maximum drawdown of an ordered portfolio-value (equity) curve.

    Convention: running_peak_t = max(values[:t+1]);
    drawdown_t = (value_t - peak_t) / peak_t (<= 0);
    result = min_t drawdown_t, a NEGATIVE fraction (or 0.0 when the
    curve never declines), e.g. [100,120,110,90,105] -> -0.25 (-25%).
    Inputs must be strictly positive finite values in chronological
    order; empty input, NaN/Inf, or non-positive values raise
    ValueError.
    """
    arr = _clean_1d(values, name="values", minimum=1)
    if np.any(arr <= 0.0):
        raise ValueError("values must be strictly positive portfolio values.")
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return float(np.min(drawdowns))


# ---------------------------------------------------------------------------
# 9. Historical Value at Risk
# ---------------------------------------------------------------------------

def historical_var(returns, confidence: float = 0.95,
                   portfolio_value: float | None = None) -> float:
    """Historical VaR at ``confidence`` (e.g. 0.95).

    Method: alpha = 1 - confidence; q = linear-interpolated historical
    quantile of ``returns`` at alpha (numpy default ``method="linear"``);
    VaR_return = -q. CONVENTION: returned as a POSITIVE loss amount in
    fractional-return units (e.g. 0.03 = 3% loss threshold); a negative
    result means the tail quantile was itself a gain. When
    ``portfolio_value`` (positive, finite notional) is given, the result
    is scaled to currency: VaR_amount = VaR_return * portfolio_value.
    This is a historical-quantile threshold, NOT a guaranteed maximum
    loss. ``confidence`` must satisfy 0 < confidence < 1; ``returns``
    needs >= 2 finite observations (larger samples give stabler tails).
    """
    arr = _clean_1d(returns, name="returns", minimum=2)
    try:
        conf = float(confidence)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"confidence must be a numeric scalar: {exc}") from exc
    if not math.isfinite(conf) or not 0.0 < conf < 1.0:
        raise ValueError(
            f"confidence must satisfy 0 < confidence < 1, got {confidence}."
        )
    alpha = 1.0 - conf
    quant = float(np.quantile(arr, alpha, method="linear"))
    var_return = -quant
    if portfolio_value is None:
        return float(var_return)
    try:
        pv = float(portfolio_value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"portfolio_value must be a numeric scalar: {exc}"
        ) from exc
    if not math.isfinite(pv) or pv <= 0.0:
        raise ValueError(
            f"portfolio_value must be finite and positive, got {portfolio_value}."
        )
    return float(var_return * pv)


# ---------------------------------------------------------------------------
# 10. Annualization helpers (explicit only)
# ---------------------------------------------------------------------------

def annualized_mean(mean_return_value: float,
                    periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualise a per-period mean: ann_mean = mean * periods_per_year.

    Default 252 (daily trading-day inputs). No compounding assumed;
    caller must pass a mean matching the input frequency.
    """
    try:
        m = float(mean_return_value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"mean_return_value must be numeric: {exc}"
        ) from exc
    if not math.isfinite(m):
        raise ValueError("mean_return_value must be finite.")
    if not isinstance(periods_per_year, (int, np.integer)) \
            or int(periods_per_year) <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    return float(m * int(periods_per_year))


def annualized_volatility(volatility: float,
                          periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualise per-period volatility: ann_vol = vol * sqrt(periods).

    Default 252 trading days (sqrt(252) for daily inputs). Volatility
    must be finite and non-negative.
    """
    try:
        v = float(volatility)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"volatility must be numeric: {exc}") from exc
    if not math.isfinite(v):
        raise ValueError("volatility must be finite.")
    if v < 0.0:
        raise ValueError(f"volatility must be non-negative, got {volatility}.")
    if not isinstance(periods_per_year, (int, np.integer)) \
            or int(periods_per_year) <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    return float(v * math.sqrt(int(periods_per_year)))
