"""Portfolio optimization engine (Task 4B).

Builds long-only, fully-invested portfolios from NUMERICAL inputs only:

    expected_returns   1-D vector of per-period expected fractional returns
    covariance_matrix  2-D square sample-covariance matrix (same ordering)
    risk_free_rate     scalar, same frequency/units as expected_returns

Objectives (exactly these three):
  * equal weight .............. w_i = 1/N (analytic, no optimizer)
  * minimum variance .......... min w^T Sigma w  (SLSQP)
  * maximum Sharpe ............ max (w^T mu - R_f)/sqrt(w^T Sigma w) (SLSQP)

Constraints for the optimized portfolios (explicit, verified post-solve):
  * long-only:        w_i >= 0   (SLSQP bounds (0, 1) per asset)
  * fully invested:   sum(w) = 1 (SLSQP equality constraint)
  * no leverage, no short selling.

This module is ONLY the optimizer. It never touches market data, dates,
rolling windows, backtesting, simulation, frontiers, recommendations,
dashboards, sentiment or news. Task 4D (rolling backtest) consumes the
structured :class:`OptimizationResult` objects returned here.

Methodology boundaries (deliberate):
  * The caller's covariance matrix is used AS GIVEN. No shrinkage,
    regularization, clipping, or diagonal noise is ever applied. A matrix
    that is asymmetric beyond tolerance or materially non-PSD raises
    ValueError instead of being silently repaired.
  * Solver output is NEVER renormalized. Constraint violations beyond
    CONSTRAINT_TOL raise ValueError.
  * All risk/return math delegates to src.portfolio_math (Task 4A
    conventions: sample estimators, positive-loss VaR, no silent
    annualization). This module does not redefine them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import portfolio_math as pm

__all__ = [
    "CONSTRAINT_TOL",
    "EQUAL_WEIGHT",
    "MINIMUM_VARIANCE",
    "MAXIMUM_SHARPE",
    "OptimizationResult",
    "OptimizationError",
    "equal_weight_portfolio",
    "minimum_variance_portfolio",
    "maximum_sharpe_portfolio",
]

EQUAL_WEIGHT = "equal_weight"
MINIMUM_VARIANCE = "minimum_variance"
MAXIMUM_SHARPE = "maximum_sharpe"

# Post-solve acceptance tolerance for sum(w) == 1 and w_i >= 0.
CONSTRAINT_TOL = 1e-8
# Symmetry acceptance for the caller-supplied covariance matrix.
_SYM_RTOL = 1e-10
_SYM_ATOL = 1e-12
# Negative-eigenvalue tolerance, scaled to matrix magnitude: eigenvalues
# below -_EIG_RTOL * max(max_eig, tiny) mean a materially indefinite matrix.
_EIG_RTOL = 1e-8
# Numerical floor for INTERMEDIATE Sharpe evaluations inside the optimizer
# only (keeps SLSQP away from 0/0 on degenerate trial points). The final
# reported Sharpe always uses the true volatility; a non-positive final
# volatility raises instead of returning a spurious ratio.
_VOL_FLOOR = 1e-12


class OptimizationError(RuntimeError):
    """Raised when optimization output is unusable.

    Covers: solver output violating constraints beyond CONSTRAINT_TOL,
    non-finite solver output, and ill-posed problems detected after the
    solve (e.g. non-positive volatility for a maximum-Sharpe solution).
    Solver NON-convergence is instead reported via
    ``OptimizationResult.success is False`` so Task 4D can inspect it.
    """


@dataclass(frozen=True)
class OptimizationResult:
    """Structured portfolio solution for Task 4D consumption.

    Fields:
      weights ............. 1-D ndarray, asset ordering preserved.
      asset_names ......... tuple of names (same order) or None.
      expected_return ..... w^T mu (NaN when mu was not supplied).
      variance ............ w^T Sigma w (NaN if not computable).
      volatility .......... sqrt(variance) (NaN if not computable).
      sharpe_ratio ........ (Rp - Rf)/vol; NaN when mu missing or vol
                            is zero/undefined (Sharpe needs vol > 0).
      objective_name ...... 'equal_weight' | 'minimum_variance' |
                            'maximum_sharpe'.
      success ............. True only if the solution is usable. Solver
                            non-convergence yields success=False.
      solver_status ....... scipy status code (0 == success; 0 with an
                            analytic message when no solver ran).
      solver_message ...... human-readable solver/analytic message.
      constraint_residual . max(|sum(w)-1|, max(0, -min(w))); worst
                            violation of the two constraint types
                            (0.0 for exact feasibility).
      risk_free_rate ...... the Rf used (None when not applicable).
    """

    weights: np.ndarray = field(repr=False)
    expected_return: float
    variance: float
    volatility: float
    sharpe_ratio: float
    objective_name: str
    success: bool
    solver_status: int
    solver_message: str
    constraint_residual: float
    asset_names: tuple | None = None
    risk_free_rate: float | None = None

    @property
    def n_assets(self) -> int:
        """Number of assets (length of weights)."""
        return int(self.weights.shape[0])

    def weights_dict(self) -> dict:
        """Weights keyed by asset name (requires asset_names)."""
        if self.asset_names is None:
            raise ValueError("weights_dict requires asset_names.")
        return {name: float(w)
                for name, w in zip(self.asset_names, self.weights)}

    def to_dict(self) -> dict:
        """Plain-dict view (Task 4D friendly; weights as list)."""
        d = {
            "weights": [float(w) for w in self.weights],
            "expected_return": float(self.expected_return),
            "variance": float(self.variance),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "objective_name": self.objective_name,
            "success": bool(self.success),
            "solver_status": int(self.solver_status),
            "solver_message": str(self.solver_message),
            "constraint_residual": float(self.constraint_residual),
            "asset_names": list(self.asset_names)
            if self.asset_names is not None else None,
            "risk_free_rate": None if self.risk_free_rate is None
            else float(self.risk_free_rate),
        }
        return d


# Input validation (fail loudly; never silently repair)

def _as_float_1d(values, *, name: str) -> np.ndarray:
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.to_numpy()
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric 1-D data: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")
    return arr


def _clean_covariance(covariance_matrix) -> np.ndarray:
    """Validate the caller's covariance matrix WITHOUT modifying it."""
    if isinstance(covariance_matrix, pd.DataFrame):
        covariance_matrix = covariance_matrix.to_numpy()
    try:
        mat = np.asarray(covariance_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"covariance_matrix must be numeric: {exc}") from exc
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            "covariance_matrix must be a square 2-D matrix, "
            f"got shape {mat.shape}."
        )
    n = mat.shape[0]
    if n == 0:
        raise ValueError("covariance_matrix must not be empty.")
    if not np.all(np.isfinite(mat)):
        raise ValueError(
            "covariance_matrix must contain only finite values (no NaN/Inf)."
        )
    if not np.allclose(mat, mat.T, rtol=_SYM_RTOL, atol=_SYM_ATOL):
        raise ValueError(
            "covariance_matrix must be symmetric; it is not symmetrized "
            "silently. Max asymmetry: "
            f"{float(np.max(np.abs(mat - mat.T))):.3e}."
        )
    eigvals = np.linalg.eigvalsh(mat)
    floor = -_EIG_RTOL * max(float(eigvals[-1]), 1e-300)
    if float(eigvals[0]) < floor:
        raise ValueError(
            "covariance_matrix is not positive semidefinite "
            f"(min eigenvalue {float(eigvals[0]):.3e}); it would produce "
            "invalid (negative) portfolio variance. No shrinkage or "
            "regularization is applied silently."
        )
    return mat


def _clean_risk_free(risk_free_rate) -> float:
    try:
        rf = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"risk_free_rate must be a numeric scalar: {exc}") from exc
    if not math.isfinite(rf):
        raise ValueError(
            f"risk_free_rate must be finite, got {risk_free_rate}.")
    return rf


def _clean_names(asset_names, n: int) -> tuple | None:
    if asset_names is None:
        return None
    names = list(asset_names)
    if len(names) != n:
        raise ValueError(
            f"asset_names length ({len(names)}) must match n_assets ({n}).")
    if any(not isinstance(x, str) for x in names):
        raise TypeError("asset_names must all be strings.")
    return tuple(names)


def _clean_x0(x0, n: int) -> np.ndarray:
    """Validate an explicit starting point (default: equal weights)."""
    if x0 is None:
        return np.full(n, 1.0 / n)
    w = _as_float_1d(x0, name="x0")
    if w.size != n:
        raise ValueError(
            f"x0 length ({w.size}) must match n_assets ({n}).")
    if np.any(w < 0.0):
        raise ValueError("x0 must satisfy long-only (all weights >= 0).")
    if abs(float(np.sum(w)) - 1.0) > 1e-6:
        raise ValueError(
            f"x0 must sum to 1 (got {float(np.sum(w)):.6f}).")
    return w


def _clean_solver_options(solver_options) -> dict:
    if solver_options is None:
        return {}
    if not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dict (or None).")
    return dict(solver_options)


def _constraint_residual(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(max(abs(float(np.sum(w)) - 1.0),
                     max(0.0, -float(np.min(w)))))


# Result assembly

def _assemble_result(weights, *, cov: np.ndarray | None,
                     mu: np.ndarray | None,
                     risk_free_rate: float | None,
                     objective_name: str, asset_names: tuple | None,
                     success: bool, solver_status: int,
                     solver_message: str,
                     strict: bool) -> OptimizationResult:
    """Build an OptimizationResult, verifying constraints on success path.

    strict=True (solver converged): constraint violations beyond
    CONSTRAINT_TOL raise OptimizationError; the output is never
    renormalized. strict=False (solver failed / analytic shortcut):
    metrics are best-effort (NaN where undefined) and the residual is
    reported for inspection. cov=None (equal weight without a matrix)
    yields NaN variance/volatility rather than a dummy-matrix value.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    residual = _constraint_residual(w)
    if strict:
        if not np.all(np.isfinite(w)):
            raise OptimizationError(
                "solver returned non-finite weights; refusing to use them.")
        if residual > CONSTRAINT_TOL:
            raise OptimizationError(
                f"solver output violates constraints (residual {residual:.3e} "
                f"> tol {CONSTRAINT_TOL:.0e}); output is not renormalized.")
    if cov is None:
        var, vol = math.nan, math.nan
    else:
        try:
            var = pm.portfolio_variance(w, cov)
            vol = pm.portfolio_volatility(w, cov)
        except ValueError:
            if strict:
                raise
            var, vol = math.nan, math.nan
    if mu is not None and math.isfinite(var):
        exp_ret = pm.portfolio_expected_return(w, mu)
    else:
        exp_ret = math.nan
    if mu is not None and risk_free_rate is not None \
            and math.isfinite(vol) and vol > 0.0:
        sharpe = pm.sharpe_ratio(exp_ret, risk_free_rate, vol)
    else:
        sharpe = math.nan
    return OptimizationResult(
        weights=w,
        expected_return=float(exp_ret),
        variance=float(var),
        volatility=float(vol),
        sharpe_ratio=float(sharpe),
        objective_name=objective_name,
        success=bool(success),
        solver_status=int(solver_status),
        solver_message=str(solver_message),
        constraint_residual=float(residual),
        asset_names=asset_names,
        risk_free_rate=None if risk_free_rate is None
        else float(risk_free_rate),
    )


def _run_slsqp(objective_fn, n: int, x0: np.ndarray,
               solver_options: dict):
    """Deterministic SLSQP solve: sum(w)=1 equality + (0,1) bounds."""
    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}
    bounds = [(0.0, 1.0)] * n
    return minimize(objective_fn, x0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options=dict(solver_options))


# 7. Equal weight (analytic; no optimizer)

def equal_weight_portfolio(n_assets: int,
                           covariance_matrix=None,
                           expected_returns=None,
                           risk_free_rate: float = 0.0,
                           asset_names=None) -> OptimizationResult:
    """Equal-weight portfolio: w_i = 1/N. No optimizer is run.

    Requires n_assets >= 1. When ``covariance_matrix`` (and optionally
    ``expected_returns``/``risk_free_rate``) is supplied, the usual risk/
    return metrics are reported; otherwise variance, volatility,
    expected return and Sharpe are NaN (no dummy matrix is invented).
    Returns success=True with solver_status=0 and a message recording
    that the solution is analytic.
    """
    if isinstance(n_assets, bool) or not isinstance(n_assets, (int, np.integer)):
        raise TypeError(
            f"n_assets must be a positive integer, got {n_assets!r}.")
    n = int(n_assets)
    if n < 1:
        raise ValueError(f"n_assets must be >= 1, got {n}.")
    names = _clean_names(asset_names, n)
    cov = None if covariance_matrix is None \
        else _clean_covariance(covariance_matrix)
    if cov is not None and cov.shape[0] != n:
        raise ValueError(
            f"covariance dimension ({cov.shape[0]}) must match "
            f"n_assets ({n}).")
    mu = None if expected_returns is None \
        else _as_float_1d(expected_returns, name="expected_returns")
    if mu is not None and mu.size != n:
        raise ValueError(
            f"expected_returns length ({mu.size}) must match "
            f"n_assets ({n}).")
    rf = _clean_risk_free(risk_free_rate)
    w = np.full(n, 1.0 / n)
    return _assemble_result(
        w, cov=cov, mu=mu, risk_free_rate=rf,
        objective_name=EQUAL_WEIGHT, asset_names=names,
        success=True, solver_status=0,
        solver_message="analytic equal-weight solution (no optimizer invoked).",
        strict=False,
    )


# Minimum variance: min w^T Sigma w

def minimum_variance_portfolio(covariance_matrix,
                               expected_returns=None,
                               risk_free_rate: float = 0.0,
                               asset_names=None, x0=None,
                               solver_options=None) -> OptimizationResult:
    """Long-only, fully-invested minimum-variance portfolio (SLSQP).

    Single-asset edge case (n == 1): the only feasible portfolio is
    weight [1.0]; it is returned directly without invoking the solver
    (documented explicitly, since a 1-D problem needs no optimization).
    ``expected_returns``/``risk_free_rate`` are optional reporting inputs
    only -- they never enter the minimum-variance objective.
    """
    cov = _clean_covariance(covariance_matrix)
    n = cov.shape[0]
    names = _clean_names(asset_names, n)
    mu = None if expected_returns is None \
        else _as_float_1d(expected_returns, name="expected_returns")
    if mu is not None and mu.size != n:
        raise ValueError(
            f"expected_returns length ({mu.size}) must match "
            f"covariance dimension ({n}).")
    rf = _clean_risk_free(risk_free_rate)
    opts = _clean_solver_options(solver_options)

    if n == 1:
        return _assemble_result(
            np.array([1.0]), cov=cov, mu=mu, risk_free_rate=rf,
            objective_name=MINIMUM_VARIANCE, asset_names=names,
            success=True, solver_status=0,
            solver_message="single-asset: only feasible weight is [1.0] "
                           "(no optimization needed).",
            strict=False,
        )

    start = _clean_x0(x0, n)
    # Smoke test: the caller's matrix must yield a valid variance already
    # at the feasible equal-weight point (raises ValueError otherwise).
    pm.portfolio_variance(start, cov)

    def objective(w):
        return pm.portfolio_variance(np.asarray(w, dtype=float), cov)

    res = _run_slsqp(objective, n, start, opts)
    if not res.success:
        return _assemble_result(
            res.x if np.all(np.isfinite(res.x)) else start,
            cov=cov, mu=mu, risk_free_rate=rf,
            objective_name=MINIMUM_VARIANCE, asset_names=names,
            success=False, solver_status=int(res.status),
            solver_message=f"SLSQP failed: {res.message}",
            strict=False,
        )
    return _assemble_result(
        res.x, cov=cov, mu=mu, risk_free_rate=rf,
        objective_name=MINIMUM_VARIANCE, asset_names=names,
        success=True, solver_status=int(res.status),
        solver_message=str(res.message), strict=True,
    )


# Maximum Sharpe: max (w^T mu - R_f) / sqrt(w^T Sigma w)

def maximum_sharpe_portfolio(expected_returns, covariance_matrix,
                             risk_free_rate: float = 0.0,
                             asset_names=None, x0=None,
                             solver_options=None) -> OptimizationResult:
    """Long-only, fully-invested maximum-Sharpe portfolio (SLSQP).

    Minimizes the negative Sharpe ratio ``-(w^T mu - R_f)/vol``.
    Single-asset edge case (n == 1): returns weight [1.0] directly.
    A non-positive final volatility raises OptimizationError (Sharpe is
    undefined there); universes containing a zero-volatility asset are
    therefore rejected loudly rather than returning a spurious ratio.
    """
    mu = _as_float_1d(expected_returns, name="expected_returns")
    cov = _clean_covariance(covariance_matrix)
    n = cov.shape[0]
    if mu.size != n:
        raise ValueError(
            f"expected_returns length ({mu.size}) must match "
            f"covariance dimension ({n}).")
    rf = _clean_risk_free(risk_free_rate)
    names = _clean_names(asset_names, n)
    opts = _clean_solver_options(solver_options)

    if n == 1:
        out = _assemble_result(
            np.array([1.0]), cov=cov, mu=mu, risk_free_rate=rf,
            objective_name=MAXIMUM_SHARPE, asset_names=names,
            success=True, solver_status=0,
            solver_message="single-asset: only feasible weight is [1.0] "
                           "(no optimization needed).",
            strict=False,
        )
        if not math.isfinite(out.volatility) or out.volatility <= 0.0:
            raise OptimizationError(
                "single-asset Sharpe is undefined (non-positive volatility).")
        return out

    start = _clean_x0(x0, n)
    pm.portfolio_variance(start, cov)  # smoke test on caller matrix
    if not (pm.portfolio_volatility(start, cov) > 0.0):
        raise ValueError(
            "degenerate universe: equal-weight volatility is not positive, "
            "so maximum Sharpe is ill-posed.")

    def neg_sharpe(w):
        w = np.asarray(w, dtype=float)
        excess = float(np.dot(w, mu)) - rf
        vol = pm.portfolio_volatility(w, cov)
        return -excess / max(vol, _VOL_FLOOR)

    res = _run_slsqp(neg_sharpe, n, start, opts)
    if not res.success:
        return _assemble_result(
            res.x if np.all(np.isfinite(res.x)) else start,
            cov=cov, mu=mu, risk_free_rate=rf,
            objective_name=MAXIMUM_SHARPE, asset_names=names,
            success=False, solver_status=int(res.status),
            solver_message=f"SLSQP failed: {res.message}",
            strict=False,
        )
    final_vol = pm.portfolio_volatility(np.asarray(res.x, dtype=float), cov)
    if not math.isfinite(final_vol) or final_vol <= 0.0:
        raise OptimizationError(
            "maximum-Sharpe solution has non-positive volatility; "
            "Sharpe is undefined (possibly a zero-volatility asset in the "
            "universe). No spurious ratio is returned.")
    return _assemble_result(
        res.x, cov=cov, mu=mu, risk_free_rate=rf,
        objective_name=MAXIMUM_SHARPE, asset_names=names,
        success=True, solver_status=int(res.status),
        solver_message=str(res.message), strict=True,
    )
