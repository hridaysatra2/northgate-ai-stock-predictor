"""Efficient frontier + Monte Carlo opportunity-space analysis (Task 4C).

Geometry of the feasible long-only portfolio set under SUPPLIED
assumptions (expected-return vector mu, covariance Sigma, risk-free
rate). This is NOT a historical performance test: no realized future
returns are computed, nothing here is profitable or "best", and the
full-sample demo outputs are illustrative only (Task 4D owns rolling
estimation and backtesting).

Components:
  * :func:`simulate_portfolios` -- ~20k Dirichlet(1) random long-only
    portfolios, fully vectorized NumPy (the SLSQP optimizer is NEVER
    called per-portfolio).
  * :func:`efficient_frontier` -- true constrained-optimization frontier:
    a sequence of minimum-variance SLSQP solves (same formulation and
    tolerances as Task 4B, plus one w^T mu = R equality each), over the
    feasible long-only return range [min(mu), max(mu)].
  * :func:`key_portfolios` -- thin wrapper returning the Task 4B
    equal-weight / minimum-variance / maximum-Sharpe solutions
    (no reimplemented optimization logic).
  * :func:`run_frontier_analysis` + :func:`main` -- explicit CLI entry
    (``python -m src.portfolio_frontier``) writing
    ``data/processed/portfolio_frontier/``. Demo assumptions come only
    from already-existing processed features and are labelled as
    in-sample illustration, never as a backtest.

Conventions inherited (not redefined): Task 4A sample estimators and
NEG_VAR_TOL float-artifact rule; Task 4B CONSTRAINT_TOL, long-only /
fully-invested constraints, and covariance validation (reused by
delegation, never duplicated or silently repaired).
"""

from __future__ import annotations

import json
import logging
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import portfolio_math as pm
from src import portfolio_optimizer as popt
from src.portfolio_math import NEG_VAR_TOL
from src.portfolio_optimizer import CONSTRAINT_TOL

__all__ = [
    "DEFAULT_N_PORTFOLIOS",
    "DEFAULT_SEED",
    "DEFAULT_N_FRONTIER_POINTS",
    "TARGET_TOL",
    "FRONTIER_DIR",
    "MonteCarloResult",
    "FrontierResult",
    "simulate_portfolios",
    "efficient_frontier",
    "key_portfolios",
    "load_illustrative_assumptions",
    "run_frontier_analysis",
    "main",
]

logger = logging.getLogger(__name__)

DEFAULT_N_PORTFOLIOS = 20_000
DEFAULT_SEED = 42
DEFAULT_N_FRONTIER_POINTS = 50
# Acceptance tolerance for achieved-vs-target frontier return.
TARGET_TOL = 1e-6
FRONTIER_DIR = Path("data/processed/portfolio_frontier")
FEATURES_DIR = Path("data/processed/features")
DEMO_RETURN_COLUMN = "log_return_1d"


# Structured results

@dataclass(frozen=True)
class MonteCarloResult:
    """Vectorized Monte Carlo opportunity-space sample.

    weights ........... (M, N) float array, rows sum to 1, all >= 0.
    expected_returns .. (M,) W @ mu.
    variances ......... (M,) diag(W @ Sigma @ W^T).
    volatilities ...... (M,) sqrt(variance).
    sharpe_ratios ..... (M,) (er - Rf)/vol; NaN where vol is zero
                        (scalar Task 4A raises there; the vectorized
                        sampler records NaN instead of aborting).
    asset_names ....... tuple (ordering preserved) or None.
    seed / n_portfolios / risk_free_rate: echo of inputs.
    """

    weights: np.ndarray = field(repr=False)
    expected_returns: np.ndarray = field(repr=False)
    variances: np.ndarray = field(repr=False)
    volatilities: np.ndarray = field(repr=False)
    sharpe_ratios: np.ndarray = field(repr=False)
    asset_names: tuple | None = None
    seed: int = DEFAULT_SEED
    n_portfolios: int = DEFAULT_N_PORTFOLIOS
    risk_free_rate: float = 0.0

    @property
    def n_assets(self) -> int:
        """Number of assets (columns of weights)."""
        return int(self.weights.shape[1])

    def to_dataframe(self) -> pd.DataFrame:
        """Tabular view: one row per portfolio, w_<name|asset_i> columns."""
        if self.asset_names is None:
            wcols = [f"w_asset_{i}" for i in range(self.n_assets)]
        else:
            wcols = [f"w_{n}" for n in self.asset_names]
        df = pd.DataFrame(np.asarray(self.weights, dtype=float),
                          columns=wcols)
        df.insert(0, "portfolio_id", np.arange(self.n_portfolios))
        df["expected_return"] = np.asarray(self.expected_returns,
                                           dtype=float)
        df["variance"] = np.asarray(self.variances, dtype=float)
        df["volatility"] = np.asarray(self.volatilities, dtype=float)
        df["sharpe_ratio"] = np.asarray(self.sharpe_ratios, dtype=float)
        return df


@dataclass(frozen=True)
class FrontierResult:
    """Constrained-optimization efficient frontier.

    One row per attempted target return (K = n_frontier_points).
    Failed points keep a NaN weight row and success=False -- never a
    fabricated portfolio. ``is_partial`` reports whether every point
    solved; check it before plotting/using the frontier.
    """

    target_returns: np.ndarray = field(repr=False)
    expected_returns: np.ndarray = field(repr=False)
    variances: np.ndarray = field(repr=False)
    volatilities: np.ndarray = field(repr=False)
    sharpe_ratios: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    success: np.ndarray = field(repr=False)
    solver_statuses: np.ndarray = field(repr=False)
    solver_messages: tuple = ()
    asset_names: tuple | None = None
    risk_free_rate: float = 0.0

    @property
    def n_points(self) -> int:
        """Number of attempted frontier points."""
        return int(self.target_returns.shape[0])

    @property
    def n_assets(self) -> int:
        """Number of assets (columns of weights)."""
        return int(self.weights.shape[1])

    @property
    def n_successful(self) -> int:
        """Number of points with success=True."""
        return int(np.sum(self.success))

    @property
    def is_partial(self) -> bool:
        """True when at least one frontier point failed to solve."""
        return bool(np.any(~np.asarray(self.success, dtype=bool)))

    def to_dataframe(self) -> pd.DataFrame:
        """Tabular view with target/achieved metrics and weight columns."""
        if self.asset_names is None:
            wcols = [f"w_asset_{i}" for i in range(self.n_assets)]
        else:
            wcols = [f"w_{n}" for n in self.asset_names]
        df = pd.DataFrame(np.asarray(self.weights, dtype=float),
                          columns=wcols)
        df.insert(0, "point_id", np.arange(self.n_points))
        df["target_return"] = np.asarray(self.target_returns, dtype=float)
        df["expected_return"] = np.asarray(self.expected_returns,
                                           dtype=float)
        df["variance"] = np.asarray(self.variances, dtype=float)
        df["volatility"] = np.asarray(self.volatilities, dtype=float)
        df["sharpe_ratio"] = np.asarray(self.sharpe_ratios, dtype=float)
        df["success"] = np.asarray(self.success, dtype=bool)
        return df


# Validation (delegates covariance rules to Task 4B; never repairs)

def _clean_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}.")
    seed = int(seed)
    if not 0 <= seed < 2 ** 32:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**32, got {seed}.")
    return seed


def _clean_count(value, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _validated_inputs(expected_returns, covariance_matrix,
                      risk_free_rate, asset_names):
    """Validate mu/Sigma/Rf/names via Task 4B conventions; return arrays.

    Delegates to ``equal_weight_portfolio`` purely as a validator: it
    enforces square/symmetric/PSD/finite covariance, 1-D finite mu with
    matching dimension, finite Rf, and name-length checks -- raising
    instead of repairing. Returns plain (mu, cov, rf, names) arrays.
    """
    mu = np.asarray(expected_returns, dtype=float)
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("expected_returns must be a non-empty 1-D vector.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("expected_returns must be finite (no NaN/Inf).")
    n = mu.size
    # Task 4B-owned validation of Sigma/Rf/names (raises, never repairs).
    popt.equal_weight_portfolio(n, covariance_matrix=covariance_matrix,
                                expected_returns=mu,
                                risk_free_rate=risk_free_rate,
                                asset_names=asset_names)
    if isinstance(covariance_matrix, pd.DataFrame):
        cov = covariance_matrix.to_numpy(dtype=float)
    else:
        cov = np.asarray(covariance_matrix, dtype=float)
    try:
        rf = float(risk_free_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"risk_free_rate must be numeric: {exc}") from exc
    names = None if asset_names is None else tuple(asset_names)
    return mu, cov, rf, names


# 2-3. Monte Carlo simulation (vectorized; no per-portfolio optimizer)

def simulate_portfolios(expected_returns, covariance_matrix,
                        risk_free_rate: float = 0.0,
                        n_portfolios: int = DEFAULT_N_PORTFOLIOS,
                        seed: int = DEFAULT_SEED,
                        asset_names=None) -> MonteCarloResult:
    """Sample ``n_portfolios`` long-only fully-invested portfolios.

    Method: Dirichlet(1, ..., 1) rows via ``default_rng(seed)`` -- the
    standard uniform-over-simplex sampler (every row w_i >= 0,
    sum(w) == 1 up to float rounding). Metrics are fully vectorized:
    ``er = W @ mu`` and ``var = einsum('ij,jk,ik->i', W, Sigma, W)``.
    Tiny negative float artifacts within Task 4A NEG_VAR_TOL are treated
    as zero; sampled Sharpe is NaN where volatility is zero (documented
    vectorized analogue of Task 4A raising for the scalar case).
    Deterministic for fixed (inputs, seed); different seeds differ.
    """
    mu, cov, rf, names = _validated_inputs(
        expected_returns, covariance_matrix, risk_free_rate, asset_names)
    m = _clean_count(n_portfolios, name="n_portfolios", minimum=1)
    seed = _clean_seed(seed)
    n = mu.size

    rng = np.random.default_rng(seed)
    if n == 1:
        weights = np.ones((m, 1))
    else:
        weights = rng.dirichlet(np.ones(n), size=m)

    er = weights @ mu
    var = np.einsum("ij,jk,ik->i", weights, cov, weights)
    bad = var < -NEG_VAR_TOL
    if np.any(bad):
        raise ValueError(
            "sampled portfolio variance is materially negative; the "
            "covariance matrix is inconsistent with PSD assumptions.")
    var = np.where(var < 0.0, 0.0, var)
    vol = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(vol > 0.0, (er - rf) / vol, np.nan)
    return MonteCarloResult(
        weights=np.asarray(weights, dtype=float),
        expected_returns=np.asarray(er, dtype=float),
        variances=np.asarray(var, dtype=float),
        volatilities=np.asarray(vol, dtype=float),
        sharpe_ratios=np.asarray(sharpe, dtype=float),
        asset_names=names, seed=seed, n_portfolios=m, risk_free_rate=rf,
    )


# 4-5/7. Efficient frontier (constrained optimization, not MC sorting)

def _min_variance_at_target(mu: np.ndarray, cov: np.ndarray,
                            target: float,
                            solver_options: dict) -> tuple:
    """SLSQP min-variance solve with sum(w)=1, w>=0, w^T mu=target.

    Starts from equal weights; on failure retries once from a one-hot
    start on the asset nearest the target (deterministic fallback that
    rescues numerically awkward endpoints). Returns
    (weights|None, status, message, retried).
    """
    n = mu.size
    constraints = ({"type": "eq",
                    "fun": lambda w: float(np.sum(w)) - 1.0},
                   {"type": "eq",
                    "fun": lambda w, t=target: float(w @ mu) - t})
    bounds = [(0.0, 1.0)] * n

    def objective(w):
        return pm.portfolio_variance(np.asarray(w, dtype=float), cov)

    starts = [np.full(n, 1.0 / n)]
    corner = np.zeros(n)
    corner[int(np.argmin(np.abs(mu - target)))] = 1.0
    if not np.allclose(corner, starts[0]):
        starts.append(corner)
    last = None
    for attempt, x0 in enumerate(starts):
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints,
                       options=dict(solver_options))
        last = res
        w = np.asarray(res.x, dtype=float)
        ok = (res.success and np.all(np.isfinite(w))
              and max(abs(float(np.sum(w)) - 1.0),
                      max(0.0, -float(np.min(w)))) <= CONSTRAINT_TOL
              and abs(float(w @ mu) - target) <= TARGET_TOL)
        if ok:
            msg = str(res.message)
            if attempt:
                msg = f"solved after deterministic corner retry: {msg}"
            return w, int(res.status), msg, bool(attempt)
    assert last is not None
    return None, int(last.status), f"SLSQP failed: {last.message}", True


def efficient_frontier(expected_returns, covariance_matrix,
                       risk_free_rate: float = 0.0,
                       n_frontier_points: int = DEFAULT_N_FRONTIER_POINTS,
                       asset_names=None,
                       solver_options=None) -> FrontierResult:
    """Minimum-variance frontier over the feasible long-only range.

    Targets are ``linspace(min(mu), max(mu), K)``: for long-only
    fully-invested portfolios the achieved return is a convex
    combination of asset returns, so this interval is exactly the
    feasible range. Each target solves min w^T Sigma w subject to
    sum(w)=1, w>=0, w^T mu=R with SLSQP (Task 4B formulation plus the
    return equality; per-point math via Task 4A). Degenerate cases:
    single asset (only [1.0] feasible) and flat mu (one Task 4B
    minimum-variance solve replicated across identical targets) are
    handled explicitly. Failed points keep NaN weights with
    success=False and ``is_partial`` reports a partial frontier -- no
    fake points are ever fabricated.
    """
    mu, cov, rf, names = _validated_inputs(
        expected_returns, covariance_matrix, risk_free_rate, asset_names)
    k = _clean_count(n_frontier_points, name="n_frontier_points",
                     minimum=2)
    if solver_options is not None and not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dict (or None).")
    opts = dict(solver_options) if solver_options is not None else {}
    n = mu.size
    lo, hi = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(lo, hi, k)

    if n == 1 or lo == hi:
        # Trivially feasible: [1.0], or one min-variance solve replicated.
        base = (np.array([1.0]) if n == 1
                else popt.minimum_variance_portfolio(
                    cov, expected_returns=mu, risk_free_rate=rf,
                    asset_names=None, solver_options=opts).weights)
        if n > 1 and not np.all(np.isfinite(base)):
            raise popt.OptimizationError(
                "base minimum-variance solve for flat-mu frontier failed.")
        wmat = np.tile(np.asarray(base, dtype=float), (k, 1))
        vols = np.array([pm.portfolio_volatility(w, cov) for w in wmat])
        vols = np.where(vols <= 0.0, np.nan, vols)
        achieved = wmat @ mu
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.where(np.isfinite(vols), (achieved - rf) / vols, np.nan)
        return FrontierResult(
            target_returns=np.asarray(targets, dtype=float),
            expected_returns=np.asarray(achieved, dtype=float),
            variances=np.array([pm.portfolio_variance(w, cov)
                                for w in wmat]),
            volatilities=np.asarray(vols, dtype=float),
            sharpe_ratios=np.asarray(sh, dtype=float),
            weights=np.asarray(wmat, dtype=float),
            success=np.ones(k, dtype=bool),
            solver_statuses=np.zeros(k, dtype=int),
            solver_messages=tuple(
                "single-asset feasible point" if n == 1
                else "flat-mu: replicated Task 4B minimum-variance solution"
                for _ in range(k)),
            asset_names=names, risk_free_rate=rf,
        )

    wmat = np.full((k, n), np.nan)
    achieved = np.full(k, np.nan)
    variances = np.full(k, np.nan)
    volatilities = np.full(k, np.nan)
    sharpes = np.full(k, np.nan)
    success = np.zeros(k, dtype=bool)
    statuses = np.zeros(k, dtype=int)
    messages: list[str] = []
    for i, target in enumerate(targets):
        w, status, message, _ = _min_variance_at_target(
            mu, cov, float(target), opts)
        statuses[i] = status
        messages.append(message)
        if w is None:
            continue
        wmat[i] = w
        achieved[i] = float(w @ mu)
        variances[i] = pm.portfolio_variance(w, cov)
        vol = pm.portfolio_volatility(w, cov)
        volatilities[i] = vol
        sharpes[i] = pm.sharpe_ratio(achieved[i], rf, vol) if vol > 0 else np.nan
        success[i] = True
    return FrontierResult(
        target_returns=np.asarray(targets, dtype=float),
        expected_returns=achieved, variances=variances,
        volatilities=volatilities, sharpe_ratios=sharpes, weights=wmat,
        success=success, solver_statuses=statuses,
        solver_messages=tuple(messages), asset_names=names,
        risk_free_rate=rf,
    )


# 6. Key portfolios (Task 4B owned; no logic duplicated)

def key_portfolios(expected_returns, covariance_matrix,
                   risk_free_rate: float = 0.0,
                   asset_names=None,
                   solver_options=None) -> dict:
    """Equal-weight / minimum-variance / maximum-Sharpe via Task 4B.

    Returns ``{"equal_weight": ..., "minimum_variance": ...,
    "maximum_sharpe": ...}`` of Task 4B OptimizationResult objects.
    """
    mu, cov, rf, names = _validated_inputs(
        expected_returns, covariance_matrix, risk_free_rate, asset_names)
    if solver_options is not None and not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dict (or None).")
    opts = dict(solver_options) if solver_options is not None else {}
    return {
        "equal_weight": popt.equal_weight_portfolio(
            mu.size, covariance_matrix=cov, expected_returns=mu,
            risk_free_rate=rf, asset_names=list(names)
            if names is not None else None),
        "minimum_variance": popt.minimum_variance_portfolio(
            cov, expected_returns=mu, risk_free_rate=rf,
            asset_names=list(names) if names is not None else None,
            solver_options=opts),
        "maximum_sharpe": popt.maximum_sharpe_portfolio(
            mu, cov, risk_free_rate=rf,
            asset_names=list(names) if names is not None else None,
            solver_options=opts),
    }


# 14-15. Explicit analysis entry point + CLI (demo data only, no backtest)

def load_illustrative_assumptions(
        features_dir: str | Path = FEATURES_DIR,
        tickers: list[str] | None = None,
        return_column: str = DEMO_RETURN_COLUMN,
        config_path: str | Path = "config.yaml") -> dict:
    """Build illustrative (mu, Sigma) from EXISTING processed features.

    Uses the ``log_return_1d`` panel across tickers (inner-joined dates),
    with sample mean (Task 4A convention: plain arithmetic mean) and
    sample covariance (ddof=1). STRICTLY an in-sample opportunity-space
    illustration: it must never be presented as a backtest, an
    out-of-sample test, or evidence of profitability. Returns dict with
    mu, cov, tickers, date_min/max, n_obs, frequency and return type.
    """
    import yaml  # local import: pyyaml is a declared project dependency

    features_dir = Path(features_dir)
    if tickers is None:
        with open(config_path, "r", encoding="utf-8") as f:
            tickers = list(yaml.safe_load(f)["universe"]["tickers"])
    frames = []
    for t in tickers:
        path = features_dir / f"{t}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"existing feature file missing: {path}")
        df = pd.read_parquet(path, columns=["date", return_column])
        df = df.dropna(subset=[return_column]).rename(columns={return_column: t})
        frames.append(df.set_index("date"))
    panel = pd.concat(frames, axis=1, join="inner").sort_index().dropna()
    if panel.shape[0] < 2:
        raise ValueError("insufficient overlapping observations for demo.")
    mu = panel.mean(axis=0).to_numpy(dtype=float)
    cov = panel.cov(ddof=1).to_numpy(dtype=float)
    return {
        "mu": mu, "cov": cov, "tickers": list(panel.columns),
        "date_min": str(panel.index.min()),
        "date_max": str(panel.index.max()),
        "n_obs": int(panel.shape[0]),
        "frequency": "daily",
        "return_definition": f"features {return_column} (log, per-day)",
        "estimator": "sample mean / sample covariance (ddof=1)",
    }


def run_frontier_analysis(output_dir: str | Path = FRONTIER_DIR,
                          n_portfolios: int = DEFAULT_N_PORTFOLIOS,
                          seed: int = DEFAULT_SEED,
                          n_frontier_points: int = DEFAULT_N_FRONTIER_POINTS,
                          risk_free_rate: float = 0.0,
                          tickers: list[str] | None = None) -> dict:
    """Run MC + frontier + key portfolios; write 4 output files.

    Demo assumptions only (see :func:`load_illustrative_assumptions`);
    writes monte_carlo_portfolios.csv, efficient_frontier.csv,
    key_portfolios.csv and frontier_metadata.json. Never trains models,
    never backtests.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = load_illustrative_assumptions(tickers=tickers)
    mu, cov, names = demo["mu"], demo["cov"], demo["tickers"]

    mc = simulate_portfolios(mu, cov, risk_free_rate=risk_free_rate,
                             n_portfolios=n_portfolios, seed=seed,
                             asset_names=names)
    fr = efficient_frontier(mu, cov, risk_free_rate=risk_free_rate,
                            n_frontier_points=n_frontier_points,
                            asset_names=names)
    keys = key_portfolios(mu, cov, risk_free_rate=risk_free_rate,
                          asset_names=names)

    mc_path = out_dir / "monte_carlo_portfolios.csv"
    fr_path = out_dir / "efficient_frontier.csv"
    key_path = out_dir / "key_portfolios.csv"
    meta_path = out_dir / "frontier_metadata.json"
    mc.to_dataframe().to_csv(mc_path, index=False)
    fr.to_dataframe().to_csv(fr_path, index=False)
    key_rows = []
    for name, res in keys.items():
        row = {"portfolio": name,
               "expected_return": res.expected_return,
               "variance": res.variance, "volatility": res.volatility,
               "sharpe_ratio": res.sharpe_ratio, "success": res.success,
               "solver_status": res.solver_status,
               "constraint_residual": res.constraint_residual}
        row.update(res.weights_dict())
        key_rows.append(row)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    metadata = {
        "task": "4C efficient frontier + Monte Carlo opportunity space",
        "not_a_backtest": True,
        "disclaimer": ("In-sample geometric illustration under supplied "
                       "sample-mean/covariance assumptions. No realized "
                       "future returns computed; no profitability claimed; "
                       "maximum Sharpe is not endorsed as best investment."),
        "tickers": names, "n_assets": len(names),
        "date_min": demo["date_min"], "date_max": demo["date_max"],
        "n_obs": demo["n_obs"], "frequency": demo["frequency"],
        "return_definition": demo["return_definition"],
        "estimator": demo["estimator"],
        "risk_free_rate": float(risk_free_rate),
        "n_portfolios": int(mc.n_portfolios), "seed": int(mc.seed),
        "n_frontier_points": int(fr.n_points),
        "n_frontier_successful": int(fr.n_successful),
        "frontier_partial": bool(fr.is_partial),
        "mc_weight_method": "Dirichlet(1,...,1) via numpy default_rng",
        "frontier_method": ("per-target SLSQP min-variance "
                            "(Task 4B formulation + w^T mu = R)"),
        "key_portfolios": {k: v.to_dict() for k, v in keys.items()},
        "files": [p.name for p in (mc_path, fr_path, key_path, meta_path)],
    }
    meta_path.write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    print("=" * 72)
    print("TASK 4C COMPLETE - opportunity space only (no backtest)")
    print(f"Outputs -> {out_dir} "
          f"(MC={mc.n_portfolios}, frontier={fr.n_successful}/{fr.n_points}, "
          f"partial={fr.is_partial})")
    print("=" * 72)
    return {"output_dir": str(out_dir), "metadata": metadata,
            "mc": mc, "frontier": fr, "key_portfolios": keys}


def main() -> dict:
    """CLI entry point: ``python -m src.portfolio_frontier``."""
    return run_frontier_analysis()


if __name__ == "__main__":
    main()
