"""Leakage-free rolling portfolio backtest engine (Task 4D).

Simulates a portfolio process where weights at each rebalance use ONLY
information strictly before the rebalance date. Compares equal-weight,
rolling minimum-variance, rolling maximum-Sharpe (all via Task 4B,
never reimplemented) against the S&P 500 benchmark (evaluation only,
never optimized, never in the weight vector).

REBALANCE CONVENTION (month-end decision, next-month application):
  * Rebalance dates are the FIRST trading day of each calendar month.
  * Estimation window = the ``lookback`` (default 252) trading days
    STRICTLY before the rebalance date, i.e. information through the
    final trading day of the previous month. Hence always
    max(estimation_dates) < rebalance_date.
  * Weights decided then are held FIXED over [rebalance_date,
    next_rebalance_date) and applied ONLY to returns on those dates.
  * January returns never determine January weights: they are set from
    data ending on the last trading day of December.

RETURN CONVENTION: stored inputs are daily LOG returns (project
features convention). Each estimation window is converted ONCE via
simple = exp(log) - 1; estimation (Task 4A sample mean/covariance) and
holding-period aggregation (w . r_simple, exact for simple returns)
both live in simple-return space. No adjusted/unadjusted mixing.

FAILURE POLICY (explicit, never fabricated): if estimation or
optimization fails at a rebalance (invalid covariance, solver
non-convergence), the previous target weights are HELD (no trade,
turnover 0 for that date, status recorded). If no previous weights
exist, the strategy holds CASH (0% daily return, turnover 0) until its
first successful rebalance. Full ``lookback`` is always required; short
windows are never silently used.

COSTS: gross performance + turnover are reported; transaction costs are
NOT modeled (no invented assumptions). Nothing here is投资 advice: the
output is a descriptive historical comparison, not a claim that any
strategy is best or will profit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import portfolio_math as pm
from src import portfolio_optimizer as popt

__all__ = [
    "UNIVERSE",
    "BENCHMARK_TICKER",
    "RESEARCH_CUTOFF",
    "DEFAULT_LOOKBACK",
    "DEFAULT_INITIAL_CAPITAL",
    "DEFAULT_RISK_FREE_RATE_ANNUAL",
    "DEFAULT_SOLVER_OPTIONS",
    "DEFAULT_SORTINO_TARGET",
    "DEFAULT_VAR_CONFIDENCE",
    "TRADING_DAYS_PER_YEAR",
    "EQUAL_WEIGHT",
    "ROLLING_MIN_VARIANCE",
    "ROLLING_MAX_SHARPE",
    "BENCHMARK",
    "STRATEGIES",
    "BacktestResult",
    "calmar_ratio",
    "run_rolling_backtest",
    "load_backtest_data",
    "write_backtest_outputs",
    "run_backtest_cli",
    "main",
]

logger = logging.getLogger(__name__)

UNIVERSE = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM", "HD", "V", "CAT",
            "KO"]
BENCHMARK_TICKER = "GSPC"
RESEARCH_CUTOFF = "2026-08-31"
DEFAULT_LOOKBACK = 252
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_RISK_FREE_RATE_ANNUAL = 0.0
DEFAULT_SORTINO_TARGET = 0.0
DEFAULT_VAR_CONFIDENCE = 0.95
TRADING_DAYS_PER_YEAR = 252
# Default SLSQP precision for the rolling optimizers. Daily-return
# variances are O(1e-5), so SciPy's default ftol=1e-6 terminates at the
# starting point; 1e-14 restores genuine convergence. This is a fixed
# numerical-scale setting (documented here and in the methodology note),
# NOT tuned to backtest performance. Task 4B methodology is unchanged.
DEFAULT_SOLVER_OPTIONS = {"ftol": 1e-14}
FEATURES_DIR = Path("data/processed/features")
BACKTEST_DIR = Path("data/processed/portfolio_backtest")

EQUAL_WEIGHT = "equal_weight"
ROLLING_MIN_VARIANCE = "rolling_min_variance"
ROLLING_MAX_SHARPE = "rolling_max_sharpe"
BENCHMARK = "sp500"
STRATEGIES = (EQUAL_WEIGHT, ROLLING_MIN_VARIANCE, ROLLING_MAX_SHARPE)


# Backtest-level Calmar ratio (Task 4A has no Calmar; kept local per spec)

def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calmar = annualized_return / abs(max_drawdown).

    ``max_drawdown`` follows the Task 4A convention (negative fraction
    or 0.0). Zero drawdown is handled explicitly: +inf when the
    annualized return is positive (return without pain), NaN otherwise
    (undefined). A positive ``max_drawdown`` input raises ValueError.
    """
    try:
        ret = float(annualized_return)
        mdd = float(max_drawdown)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"calmar inputs must be numeric: {exc}") from exc
    if not math.isfinite(ret) or not math.isfinite(mdd):
        raise ValueError("calmar inputs must be finite.")
    if mdd > 0.0:
        raise ValueError(
            f"max_drawdown must follow the negative-fraction convention, "
            f"got {mdd}.")
    if mdd == 0.0:
        return math.inf if ret > 0.0 else math.nan
    return ret / abs(mdd)


# Result container

@dataclass
class BacktestResult:
    """Rolling-backtest outputs (attributes are documented in the writer).

    dates ............ DatetimeIndex of holding-period trading days.
    daily_returns .... {name: (T,) simple-return arrays}, incl. benchmark.
    values ........... {name: (T,) value curves from initial_capital}.
    rebalance_records  list of dicts: rebalance_date, estimation_start,
                       estimation_end, strategy, weights (dict), status,
                       turnover.
    metrics .......... {name: {metric: value}} for strategies + benchmark.
    summary .......... config + period + failure counts + disclaimers.
    tickers .......... universe ordering used in weight vectors.
    """

    dates: pd.DatetimeIndex
    daily_returns: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    rebalance_records: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    tickers: list = field(default_factory=list)


# Validation helpers

def _clean_panel(log_returns: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(log_returns, pd.DataFrame):
        raise TypeError("log_returns must be a pandas DataFrame.")
    if not isinstance(log_returns.index, pd.DatetimeIndex):
        raise TypeError("log_returns must have a DatetimeIndex.")
    if not log_returns.index.is_unique:
        raise ValueError("log_returns index has duplicate dates.")
    if not log_returns.index.is_monotonic_increasing:
        raise ValueError("log_returns index must be sorted ascending.")
    missing = [t for t in tickers if t not in log_returns.columns]
    if missing:
        raise ValueError(f"universe tickers missing from panel: {missing}.")
    panel = log_returns[tickers].sort_index()
    if panel.empty:
        raise ValueError("return panel is empty.")
    if not np.all(np.isfinite(panel.to_numpy(dtype=float))):
        raise ValueError(
            "return panel must contain only finite values (no NaN/Inf); "
            "missing returns are never filled or set to zero.")
    return panel


def _clean_benchmark(bench, dates: pd.DatetimeIndex) -> pd.Series | None:
    if bench is None:
        return None
    s = pd.Series(bench, index=dates) if not isinstance(bench, pd.Series) \
        else bench.reindex(dates)
    if s.isna().any():
        raise ValueError("benchmark lacks coverage for backtest dates.")
    if not np.all(np.isfinite(s.to_numpy(dtype=float))):
        raise ValueError("benchmark must contain only finite values.")
    return s


# Core rolling engine (pure function of the supplied panel)

def run_rolling_backtest(log_returns: pd.DataFrame,
                         benchmark_log_returns=None,
                         tickers: list[str] | None = None,
                         lookback: int = DEFAULT_LOOKBACK,
                         initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                         risk_free_rate_annual: float =
                         DEFAULT_RISK_FREE_RATE_ANNUAL,
                         sortino_target: float = DEFAULT_SORTINO_TARGET,
                         var_confidence: float = DEFAULT_VAR_CONFIDENCE,
                         solver_options=None,
                         start=None, end=None) -> BacktestResult:
    """Run the leakage-free monthly rolling backtest on a return panel.

    ``log_returns``: DataFrame of daily LOG returns indexed by trading
    date (must predate analysis; the engine never looks ahead).
    ``benchmark_log_returns``: Series of daily LOG benchmark returns
    (evaluation only). ``start``/``end`` clip the panel (inclusive).
    See module docstring for rebalance, failure, cost and return
    conventions.
    """
    tickers = list(UNIVERSE) if tickers is None else list(tickers)
    if len(tickers) < 1:
        raise ValueError("tickers must be non-empty.")
    if isinstance(lookback, bool) or not isinstance(lookback, int) \
            or lookback < 2:
        raise ValueError(f"lookback must be an integer >= 2, got {lookback}.")
    if not math.isfinite(float(initial_capital)) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive.")
    rf_ann = float(risk_free_rate_annual)
    if not math.isfinite(rf_ann):
        raise ValueError("risk_free_rate_annual must be finite.")
    rf_daily = rf_ann / TRADING_DAYS_PER_YEAR
    var_conf = float(var_confidence)
    if not math.isfinite(var_conf) or not 0.0 < var_conf < 1.0:
        raise ValueError("var_confidence must satisfy 0 < c < 1.")
    opts = dict(solver_options) if solver_options is not None \
        else dict(DEFAULT_SOLVER_OPTIONS)

    panel = _clean_panel(log_returns, tickers)
    if start is not None:
        panel = panel.loc[pd.Timestamp(start):]
    if end is not None:
        panel = panel.loc[:pd.Timestamp(end)]
    if len(panel) <= lookback:
        raise ValueError(
            f"insufficient observations ({len(panel)}) for lookback "
            f"{lookback}: no rebalance possible.")
    bench = _clean_benchmark(benchmark_log_returns, panel.index)

    dates = panel.index
    simple = np.expm1(panel.to_numpy(dtype=float))  # (T, N) simple returns
    bench_simple = np.expm1(bench.to_numpy(dtype=float)) \
        if bench is not None else None
    n_assets = len(tickers)

    # Rebalance dates: first trading day of each calendar month with a
    # full lookback strictly before it.
    months = dates.to_period("M")
    firsts = pd.Series(np.arange(len(dates)), index=dates).groupby(months)\
        .head(1)
    rebalance_positions = [int(p) for p in firsts if int(p) >= lookback]
    if not rebalance_positions:
        raise ValueError(
            "no eligible rebalance date: full lookback never available.")

    names = tickers
    eq_w = np.full(n_assets, 1.0 / n_assets)
    prev = {s: None for s in STRATEGIES}
    records: list[dict] = []
    ret_series: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    ret_series[BENCHMARK] = []
    out_dates: list = []

    for k, pos in enumerate(rebalance_positions):
        reb_date = dates[pos]
        window = simple[pos - lookback:pos]  # strictly before reb_date
        est_start, est_end = dates[pos - lookback], dates[pos - 1]
        mu = np.array([pm.mean_return(window[:, j])
                       for j in range(n_assets)])
        # Task 4A sample covariance on the SAME trailing window only.
        try:
            cov = pm.covariance_matrix(window)
            cov_error = None
        except ValueError as exc:
            cov, cov_error = None, str(exc)

        new_w: dict[str, np.ndarray | None] = {}
        status: dict[str, str] = {}
        # --- equal weight never optimizes; assignment cannot fail ---
        new_w[EQUAL_WEIGHT] = eq_w.copy()
        status[EQUAL_WEIGHT] = "ok"
        # --- rolling optimizers (Task 4B directly) ---
        for strat, fn in (
                (ROLLING_MIN_VARIANCE, "minvar"),
                (ROLLING_MAX_SHARPE, "maxsharpe")):
            if cov_error is not None:
                new_w[strat], status[strat] = None, f"failed: {cov_error}"
                continue
            try:
                if fn == "minvar":
                    res = popt.minimum_variance_portfolio(
                        cov, expected_returns=mu,
                        risk_free_rate=rf_daily, solver_options=opts)
                else:
                    res = popt.maximum_sharpe_portfolio(
                        mu, cov, risk_free_rate=rf_daily,
                        solver_options=opts)
                if not res.success:
                    raise popt.OptimizationError(
                        f"solver reported failure: {res.solver_message}")
                new_w[strat], status[strat] = \
                    np.asarray(res.weights, dtype=float), "ok"
            except (ValueError, popt.OptimizationError, RuntimeError) as exc:
                new_w[strat] = None
                status[strat] = f"failed: {exc}"

        turn: dict[str, float] = {}
        for strat in STRATEGIES:
            if new_w[strat] is None:  # hold previous (or cash); no trade
                if prev[strat] is None:
                    turn[strat] = 0.0
                    status[strat] += " | holding cash (no position)"
                else:
                    new_w[strat] = prev[strat].copy()
                    turn[strat] = 0.0
                    status[strat] += " | holding previous weights"
            elif prev[strat] is None:  # initial purchase from cash
                turn[strat] = float(np.sum(np.abs(new_w[strat])))
            else:
                turn[strat] = float(np.sum(np.abs(new_w[strat]
                                                  - prev[strat])))
            prev[strat] = None if new_w[strat] is None else \
                new_w[strat].copy()
            records.append({
                "rebalance_date": reb_date,
                "estimation_start": est_start,
                "estimation_end": est_end,
                "strategy": strat,
                "weights": {a: float(w)
                            for a, w in zip(names, new_w[strat])}
                if new_w[strat] is not None else None,
                "status": status[strat],
                "turnover": turn[strat],
            })

        # --- holding period: fixed weights on strictly later returns ---
        end_pos = rebalance_positions[k + 1] if k + 1 < len(
            rebalance_positions) else len(dates)
        for t in range(pos, end_pos):
            out_dates.append(dates[t])
            for strat in STRATEGIES:
                w = prev[strat]
                ret_series[strat].append(
                    0.0 if w is None else float(w @ simple[t]))
            if bench_simple is not None:
                ret_series[BENCHMARK].append(float(bench_simple[t]))

    out_idx = pd.DatetimeIndex(out_dates)
    daily = {k: np.array(v) for k, v in ret_series.items()
             if k in STRATEGIES or bench_simple is not None}
    values = {}
    for k, r in daily.items():
        v = np.empty_like(r)
        v[0] = float(initial_capital) * (1.0 + r[0])
        for i in range(1, len(r)):
            v[i] = v[i - 1] * (1.0 + r[i])
        values[k] = v

    metrics = {}
    names_all = [s for s in STRATEGIES] + \
        ([BENCHMARK] if bench_simple is not None else [])
    for k in names_all:
        r, v = daily[k], values[k]
        n = len(r)
        cum = float(v[-1] / float(initial_capital) - 1.0)
        ann_geo = float((v[-1] / float(initial_capital))
                        ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
        ann_vol = pm.annualized_volatility(pm.return_volatility(r))
        try:
            sharpe = pm.sharpe_ratio(float(np.mean(r))
                                     * TRADING_DAYS_PER_YEAR,
                                     rf_ann, ann_vol)
        except ValueError:
            # degenerate zero-volatility series (e.g. all-cash): Sharpe
            # undefined -> NaN rather than aborting the backtest.
            sharpe = math.nan
        try:
            sortino = pm.sortino_ratio(r, float(sortino_target))
        except ValueError:
            # zero downside deviation: Sortino undefined -> NaN.
            sortino = math.nan
        maxdd = pm.maximum_drawdown(v)
        metrics[k] = {
            "cumulative_return": cum,
            "annualized_return_geo": ann_geo,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "maximum_drawdown": maxdd,
            "calmar_ratio": calmar_ratio(ann_geo, maxdd),
            "historical_var_daily": pm.historical_var(r, var_conf),
            "total_turnover": float(sum(
                rec["turnover"] for rec in records
                if rec["strategy"] == k)),
            "n_observations": n,
        }

    n_fail = sum(1 for rec in records
                 if rec["status"] != "ok")
    summary = {
        "lookback": lookback,
        "solver_options": opts,
        "initial_capital": float(initial_capital),
        "risk_free_rate_annual": rf_ann,
        "risk_free_rate_daily": rf_daily,
        "sortino_target_daily": float(sortino_target),
        "var_confidence": var_conf,
        "var_convention": "historical quantile, positive loss, daily horizon",
        "annualization": f"{TRADING_DAYS_PER_YEAR} trading days",
        "costs": "gross performance; transaction costs NOT modeled",
        "requested_start": str(panel.index[0].date()),
        "requested_end": str(panel.index[-1].date()),
        "actual_first_date": str(out_idx[0].date()),
        "actual_last_date": str(out_idx[-1].date()),
        "n_rebalance_dates": len(rebalance_positions),
        "n_trading_observations": len(out_idx),
        "n_failed_rebalances": n_fail,
        "strategies": list(STRATEGIES),
        "benchmark": BENCHMARK if bench_simple is not None else None,
        "disclaimer": ("Descriptive historical comparison only. No strategy "
                       "is claimed best, profitable in future, or profitable "
                       "after costs."),
    }
    return BacktestResult(dates=out_idx, daily_returns=daily,
                          values=values, rebalance_records=records,
                          metrics=metrics, summary=summary, tickers=tickers)


# Real-data loader (existing features only; clipped to research cutoff)

def load_backtest_data(
        features_dir: str | Path = FEATURES_DIR,
        tickers: list[str] | None = None,
        benchmark_ticker: str = BENCHMARK_TICKER,
        cutoff: str = RESEARCH_CUTOFF) -> dict:
    """Assemble the aligned daily LOG-return panel from existing features.

    Inner-joins universe + benchmark ``log_return_1d`` series, drops any
    residual NaN loudly (never filled), sorts, and clips to ``cutoff``
    inclusive. Returns dict(panel, benchmark, tickers, dates info).
    """
    tickers = list(UNIVERSE) if tickers is None else list(tickers)
    features_dir = Path(features_dir)
    frames = {}
    for t in tickers + [benchmark_ticker]:
        path = features_dir / f"{t}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"existing feature file missing: {path}")
        df = pd.read_parquet(path, columns=["date", "log_return_1d"])
        frames[t] = df.dropna(subset=["log_return_1d"])\
            .set_index("date")["log_return_1d"].rename(t)
    joint = pd.concat(frames, axis=1, join="inner").sort_index()
    joint = joint.loc[:pd.Timestamp(cutoff)]
    if joint.empty:
        raise ValueError("no observations within research cutoff.")
    if joint.isna().any().any():
        raise ValueError(
            "residual NaN after alignment; refusing to fill forward.")
    panel = joint[tickers]
    return {"panel": panel, "benchmark": joint[benchmark_ticker],
            "tickers": tickers, "date_min": str(panel.index.min().date()),
            "date_max": str(panel.index.max().date()),
            "n_obs": int(len(panel))}


# Outputs (CSVs + JSON + methodology note)

METHODOLOGY_TEMPLATE = """# Backtest Methodology (Task 4D)

## What this is
A descriptive, leakage-free historical comparison of three long-only,
fully-invested monthly-rebalanced equity portfolios against the S&P 500.
It is NOT investment advice, NOT a profitability claim, and no strategy
is endorsed as best.

## Rebalance convention (month-end decision, next-month application)
- Rebalance dates are the FIRST trading day of each calendar month.
- Estimation window: the {lookback} trading days STRICTLY before the
  rebalance date (data through the prior month-end), so
  max(estimation_dates) < rebalance_date always holds.
- Weights are held FIXED over [rebalance_date, next_rebalance_date) and
  applied only to returns on those dates. January returns never set
  January weights.
- First rebalance requires the full {lookback}-observation lookback;
  earlier months are skipped, never estimated on short windows.

## Estimation
- Log returns converted once per window via simple = exp(log) - 1.
- Expected returns: Task 4A arithmetic sample mean of the window.
- Covariance: Task 4A sample covariance (ddof=1) of the SAME window.
- No shrinkage, no regularization, no filling, no silent repair.

## Strategies
- equal_weight: 1/N every month (Task 4B analytic).
- rolling_min_variance: Task 4B minimum-variance on (mu_t, Sigma_t).
- rolling_max_sharpe: Task 4B maximum-Sharpe on (mu_t, Sigma_t, Rf).
- sp500 (GSPC): benchmark, evaluation only, never optimized/weighted.
- SLSQP solver options: {solver_options} (fixed numerical-scale precision
  for daily variances; Task 4B methodology unchanged, not tuned).

## Failure policy
Failed estimation/optimization holds previous weights (no trade,
turnover 0, status recorded); with no previous weights the strategy
holds cash at 0% until its first success.

## Holding aggregation and costs
- Daily portfolio simple return = w . r_simple (exact for simple
  returns); value compounds multiplicatively from {initial_capital}.
- Turnover_t = sum |new - previous| on rebalance TARGET weights
  (initial purchase from cash = 1.0; NOT divided by 2).
- Gross performance reported; transaction costs NOT modeled.

## Metrics (per strategy and benchmark)
- cumulative_return, annualized_return_geo = (V/V0)^(252/n) - 1
- annualized_volatility = Task 4A daily sample vol x sqrt(252)
- sharpe_ratio = Task 4A on (252 x daily mean, annual Rf, ann vol)
- sortino_ratio = Task 4A on daily simple returns, target {sortino_target}
- maximum_drawdown = Task 4A negative fraction of the value curve
- calmar_ratio = annualized_return_geo / |max drawdown|
  (+inf if return > 0 with zero drawdown, else NaN)
- historical_var_daily = Task 4A positive-loss quantile at {var_confidence}
- total_turnover = sum of per-rebalance turnover
- risk_free_rate_annual = {risk_free_rate_annual} (daily = annual / 252)

## Period
- Research cutoff {cutoff} (inclusive); no later data used.
- Requested {requested_start} .. {requested_end};
  actual {actual_first_date} .. {actual_last_date};
  {n_rebalance_dates} rebalances, {n_trading_observations} trading days,
  {n_failed_rebalances} failed rebalance records.
"""


def write_backtest_outputs(result: BacktestResult,
                           output_dir: str | Path = BACKTEST_DIR,
                           config: dict | None = None) -> dict:
    """Write the 7 backtest output files; return path mapping."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = result.dates.strftime("%Y-%m-%d")

    values_df = pd.DataFrame({"date": date_str})
    for k, v in result.values.items():
        values_df[k] = v
    returns_df = pd.DataFrame({"date": date_str})
    for k, r in result.daily_returns.items():
        returns_df[k] = r
    w_rows = [{"rebalance_date": rec["rebalance_date"].strftime("%Y-%m-%d"),
               "estimation_start": rec["estimation_start"].strftime("%Y-%m-%d"),
               "estimation_end": rec["estimation_end"].strftime("%Y-%m-%d"),
               "strategy": rec["strategy"], "asset": a, "weight": w,
               "status": rec["status"]}
              for rec in result.rebalance_records
              for a, w in (rec["weights"].items()
                           if rec["weights"] is not None else [])]
    t_rows = [{"rebalance_date": rec["rebalance_date"].strftime("%Y-%m-%d"),
               "strategy": rec["strategy"], "turnover": rec["turnover"],
               "status": rec["status"]}
              for rec in result.rebalance_records]
    metrics_df = pd.DataFrame(
        [{"strategy": k, **v} for k, v in result.metrics.items()])

    paths = {
        "strategy_metrics.csv": metrics_df,
        "daily_portfolio_values.csv": values_df,
        "daily_portfolio_returns.csv": returns_df,
        "rebalance_weights.csv": pd.DataFrame(w_rows),
        "rebalance_turnover.csv": pd.DataFrame(t_rows),
    }
    for name, df in paths.items():
        df.to_csv(out_dir / name, index=False)
    summary = dict(result.summary)
    summary["metrics"] = result.metrics
    if config:
        summary["run_config"] = config
    (out_dir / "backtest_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (out_dir / "BACKTEST_METHODOLOGY.md").write_text(
        METHODOLOGY_TEMPLATE.format(
            lookback=result.summary["lookback"],
            solver_options=result.summary["solver_options"],
            initial_capital=result.summary["initial_capital"],
            sortino_target=result.summary["sortino_target_daily"],
            var_confidence=result.summary["var_confidence"],
            risk_free_rate_annual=result.summary["risk_free_rate_annual"],
            cutoff=RESEARCH_CUTOFF,
            requested_start=result.summary["requested_start"],
            requested_end=result.summary["requested_end"],
            actual_first_date=result.summary["actual_first_date"],
            actual_last_date=result.summary["actual_last_date"],
            n_rebalance_dates=result.summary["n_rebalance_dates"],
            n_trading_observations=result.summary[
                "n_trading_observations"],
            n_failed_rebalances=result.summary["n_failed_rebalances"]),
        encoding="utf-8")
    return {k: str(out_dir / k) for k in
            list(paths) + ["backtest_summary.json",
                           "BACKTEST_METHODOLOGY.md"]}


# CLI (project-consistent run_* + main entry points)

def run_backtest_cli(output_dir: str | Path = BACKTEST_DIR,
                     start=None, end=None,
                     lookback: int = DEFAULT_LOOKBACK,
                     initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                     risk_free_rate_annual: float =
                     DEFAULT_RISK_FREE_RATE_ANNUAL,
                     sortino_target: float = DEFAULT_SORTINO_TARGET,
                     var_confidence: float = DEFAULT_VAR_CONFIDENCE,
                     tickers: list[str] | None = None) -> dict:
    """CLI pipeline: existing features -> rolling backtest -> 7 files."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    if end is not None and pd.Timestamp(end) > pd.Timestamp(RESEARCH_CUTOFF):
        raise ValueError(
            f"end {end} exceeds research cutoff {RESEARCH_CUTOFF}.")
    data = load_backtest_data(tickers=tickers)
    result = run_rolling_backtest(
        data["panel"], data["benchmark"], tickers=data["tickers"],
        lookback=lookback, initial_capital=initial_capital,
        risk_free_rate_annual=risk_free_rate_annual,
        sortino_target=sortino_target, var_confidence=var_confidence,
        start=start, end=end if end is not None else RESEARCH_CUTOFF)
    config = {"start": start, "end": end, "lookback": lookback,
              "initial_capital": initial_capital,
              "risk_free_rate_annual": risk_free_rate_annual,
              "sortino_target": sortino_target,
              "var_confidence": var_confidence,
              "tickers": data["tickers"]}
    paths = write_backtest_outputs(result, output_dir, config=config)
    print("=" * 72)
    print("TASK 4D COMPLETE - descriptive historical comparison "
          "(no recommendations)")
    for k, v in result.metrics.items():
        print(f"  {k}: cum={v['cumulative_return']:.4f} "
              f"ann={v['annualized_return_geo']:.4f} "
              f"vol={v['annualized_volatility']:.4f} "
              f"sharpe={v['sharpe_ratio']:.4f} "
              f"maxdd={v['maximum_drawdown']:.4f}")
    print(f"Outputs -> {output_dir} "
          f"({result.summary['n_rebalance_dates']} rebalances, "
          f"{result.summary['n_failed_rebalances']} failed)")
    print("=" * 72)
    return {"result": result, "paths": paths}


def main(argv=None) -> dict:
    """CLI entry point: ``python -m src.portfolio_backtest [options]``."""
    parser = argparse.ArgumentParser(
        description="Leakage-free rolling portfolio backtest (Task 4D).")
    parser.add_argument("--output-dir", default=str(BACKTEST_DIR))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--initial-capital", type=float,
                        default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--risk-free-rate-annual", type=float,
                        default=DEFAULT_RISK_FREE_RATE_ANNUAL)
    parser.add_argument("--sortino-target", type=float,
                        default=DEFAULT_SORTINO_TARGET)
    parser.add_argument("--var-confidence", type=float,
                        default=DEFAULT_VAR_CONFIDENCE)
    parser.add_argument("--tickers", default=None,
                        help="comma-separated tickers (default: 10-stock "
                             "Northgate universe)")
    args = parser.parse_args(argv)
    tickers = args.tickers.split(",") if args.tickers else None
    return run_backtest_cli(
        output_dir=args.output_dir, start=args.start, end=args.end,
        lookback=args.lookback, initial_capital=args.initial_capital,
        risk_free_rate_annual=args.risk_free_rate_annual,
        sortino_target=args.sortino_target,
        var_confidence=args.var_confidence, tickers=tickers)


if __name__ == "__main__":
    main()
