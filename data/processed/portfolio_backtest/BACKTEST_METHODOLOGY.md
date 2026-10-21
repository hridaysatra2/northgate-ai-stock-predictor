# Backtest Methodology (Task 4D)

## What this is
A descriptive, leakage-free historical comparison of three long-only,
fully-invested monthly-rebalanced equity portfolios against the S&P 500.
It is NOT investment advice, NOT a profitability claim, and no strategy
is endorsed as best.

## Rebalance convention (month-end decision, next-month application)
- Rebalance dates are the FIRST trading day of each calendar month.
- Estimation window: the 252 trading days STRICTLY before the
  rebalance date (data through the prior month-end), so
  max(estimation_dates) < rebalance_date always holds.
- Weights are held FIXED over [rebalance_date, next_rebalance_date) and
  applied only to returns on those dates. January returns never set
  January weights.
- First rebalance requires the full 252-observation lookback;
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
- SLSQP solver options: {'ftol': 1e-14} (fixed numerical-scale precision
  for daily variances; Task 4B methodology unchanged, not tuned).

## Failure policy
Failed estimation/optimization holds previous weights (no trade,
turnover 0, status recorded); with no previous weights the strategy
holds cash at 0% until its first success.

## Holding aggregation and costs
- Daily portfolio simple return = w . r_simple (exact for simple
  returns); value compounds multiplicatively from 100000.0.
- Turnover_t = sum |new - previous| on rebalance TARGET weights
  (initial purchase from cash = 1.0; NOT divided by 2).
- Gross performance reported; transaction costs NOT modeled.

## Metrics (per strategy and benchmark)
- cumulative_return, annualized_return_geo = (V/V0)^(252/n) - 1
- annualized_volatility = Task 4A daily sample vol x sqrt(252)
- sharpe_ratio = Task 4A on (252 x daily mean, annual Rf, ann vol)
- sortino_ratio = Task 4A on daily simple returns, target 0.0
- maximum_drawdown = Task 4A negative fraction of the value curve
- calmar_ratio = annualized_return_geo / |max drawdown|
  (+inf if return > 0 with zero drawdown, else NaN)
- historical_var_daily = Task 4A positive-loss quantile at 0.95
- total_turnover = sum of per-rebalance turnover
- risk_free_rate_annual = 0.0 (daily = annual / 252)

## Period
- Research cutoff 2026-08-31 (inclusive); no later data used.
- Requested 2016-03-31 .. 2026-08-28;
  actual 2017-04-03 .. 2026-08-28;
  113 rebalances, 2364 trading days,
  0 failed rebalance records.
