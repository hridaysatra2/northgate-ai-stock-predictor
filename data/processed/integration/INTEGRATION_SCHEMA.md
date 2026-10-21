# Integration Schema (Task 4F)

Measured from actual repository artifacts on 2026-09-27. Column names
are literal. Dates are calendar trading dates (`YYYY-MM-DD`).

## 1. Feature datasets — `data/processed/features/<TICKER>.parquet`

- Key columns: `date` (datetime64, daily, 2016-03-31 … 2026-08-28 for
  equities), `ticker`, `log_return_1d`, lags, SMAs, momentum,
  `vol_5d[_ann]`, `vol_20d[_ann]`, `vol_60d[_ann]`, volume features,
  `day_of_week`, `month`, `month_end`, `sp500_*`, `vix_*`,
  `target_log_return_1d`.
- Date convention: row `t` describes information available at the close
  of trading day `t`. Rolling features use trailing windows only
  (`min_periods=window`, rows `t-w+1..t`); `vol_20d_ann` at row `t` is
  therefore point-in-time safe.
- Return convention: **log returns** in, **log-return target** out.
  `target_log_return_1d = log(adj[t+1]/adj[t])` via `shift(-1)`; the
  last raw row is dropped. Verified: features `AAPL.target` equals
  model `actual_return` to ~1e-17 on matching dates.
- Live use: point-in-time safe per row, but files are frozen at the
  2026-08-28 research cutoff — no live feed exists (see §8D).
- Research only. Limitation: VIX file has no supervised target.

## 2. Classical predictions — `data/processed/models/predictions.csv`

- Key columns: `date, ticker, model, actual_return, predicted_return`
  (31,440 rows = 524 dates × 10 tickers × 6 models:
  `random_walk, linear_regression, ridge, random_forest, xgboost, svr`).
- Date convention: row date `t` = prediction timestamp; both returns
  cover the horizon **t → t+1** (verified: `actual_return` equals the
  feature target on date `t`, not the contemporaneous return).
- Return convention: **log returns** (both columns).
- Research/evaluation only: test period 2024-07-29 … 2026-08-28
  (n_test=524, 1-row embargo before it). Must never be presented as
  live forecasts. Limitation: no `current_price` column (joinable from
  `data/raw/market/<T>.parquet: adj_close` on date, documented).

## 3. Classical evaluation — `data/processed/evaluation/`

- `evaluation_metrics.csv`: `ticker, model, rmse, mae, mape, r2,
  directional_accuracy, metric_source, n_test` (`metric_source =
  predictions`, i.e. final test metrics).
- `best_model_by_ticker.csv`: `ticker, best_model, rmse, ...` —
  **test-selected**; prohibited as a live model-selection mechanism.
- `model_cv_results.csv` / `selected_hyperparameters.csv` (under
  `models/`): dev-set diagnostics (`dev_cv_rmse`, `n_dev=2094`,
  `n_test=524`); no live-selection rule is defined on them.
- Research only.

## 4. DL artifacts — `data/processed/deep_learning[_training]/`

- `dataset_summary.csv`: per ticker `lookback=30`, chronological
  70/15/15 split on target rows; test span **2025-02-04 … 2026-08-28**.
- `dl_predictions.csv`: `ticker, model, date, actual_return,
  predicted_return` (bilstm/gru/lstm/transformer; **log returns**,
  horizon t → t+1, same target definition).
- `dl_metrics.csv`: `ticker, model, rmse, mae, mape, r2,
  directional_accuracy, best_epoch, best_validation_loss, seed`.
- Research only. Limitation: no per-ticker best-model file; same
  test-selection prohibition applies.

## 5. Portfolio backtest — `data/processed/portfolio_backtest/`

- `rebalance_weights.csv`: `rebalance_date, estimation_start,
  estimation_end, strategy, asset, weight, status` (3,390 rows =
  113 × 3 × 10). Verified: 252-day lookback, `estimation_end <
  rebalance_date` always, monthly first-trading-day holding.
- `rebalance_turnover.csv`: `rebalance_date, strategy, turnover,
  status`. `strategy_metrics.csv`: per strategy/benchmark
  `cumulative_return, annualized_return_geo, annualized_volatility,
  sharpe_ratio, sortino_ratio, maximum_drawdown, calmar_ratio,
  historical_var_daily, total_turnover, n_observations`.
- `daily_portfolio_values.csv` / `daily_portfolio_returns.csv`:
  `date` + strategy/benchmark columns (**simple returns**, values
  compound from 100,000; gross, no costs).
- `backtest_summary.json`: config + actual period 2017-04-03 …
  2026-08-28, 113 rebalances, 0 failed.
- GSPC appears only as benchmark columns, never as a weight asset.
- Research only (gross historical comparison, not live expectation).

## 6. Frontier — `data/processed/portfolio_frontier/`

- `monte_carlo_portfolios.csv`, `efficient_frontier.csv`,
  `key_portfolios.csv`, `frontier_metadata.json`
  (`not_a_backtest: true`; in-sample illustration under sample
  mean/covariance assumptions). Research only.

## 7. Recommendation engine — `src/recommendation.py` I/O

- Input (explicit caller forecast): `ticker, as_of_date,
  current_price, predicted_log_return, model_name,
  annualized_volatility` (+ optional Sharpe/Sortino/drawdown/VaR,
  sentiment fields defaulting to unavailable).
- Output: 19 columns (see `OUTPUT_COLUMNS`); conversions
  `simple = exp(log)−1`, `price = current × exp(log)`; BUY/HOLD/SELL
  by the exact five-branch rule; reason strings fixed.
- Live-capable as pure functions, but currently blocked on live
  inputs (no forecast provider, no live risk feed — see audit §8).

## 8. Live vs research classification

- **A (live-capable code, no live data):** `src/recommendation.py`,
  `src/integration.py` (this task) as pure validation/plumbing.
- **B (historical research only):** all feature/prediction/evaluation/
  backtest/frontier artifacts listed above.
- **C (synthetic/demo only):** `demo_recommendations_SYNTHETIC.csv`
  (`DEMO_*` tickers, `synthetic_demo` model).
- **D (unavailable):** historical sentiment scores; live price/risk
  feeds; a leakage-free live forecast provider.
