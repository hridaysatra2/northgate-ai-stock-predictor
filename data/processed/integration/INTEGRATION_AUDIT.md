# Integration Audit (Task 4F)

## Timestamp chain (verified against artifacts, not just docs)

- **Features (t):** row `t` holds trailing-only features (data ≤ close
  `t`) plus `target = log(adj[t+1]/adj[t])` (`shift(-1)` in
  `src/features.py:build_target`). Last raw row dropped.
- **Forecast (t → t+1):** classical and DL prediction rows dated `t`
  forecast the return realized over `(t, t+1]`. Proven: joining
  `predictions.csv` to features on `(date, ticker)` reproduces
  `actual_return == target_log_return_1d` to ~1e-17, while the
  contemporaneous `log_return_1d` differs materially. A forecast made
  at `t` therefore uses no information after `t` **iff** its feature
  row uses none — true by trailing-window construction.
- **Splits:** classical test 2024-07-29 … 2026-08-28 (n=524, 1-row
  embargo in `models.py` so no dev target consumes a test price); DL
  test 2025-02-04 … 2026-08-28 (chronological 70/15/15 on target rows).
- **Risk (4E):** caller-supplied `annualized_volatility` must be known
  at `as_of_date`. The clean boundary is a Risk Calculator reading
  trailing `vol_20d_ann`/`vol_60d_ann` from the features row of
  `as_of_date` (rows `t-w+1..t`, point-in-time safe). Never a
  full-history statistic presented as current.
- **Backtest:** 252-day estimation window strictly before each
  month-end decision; weights held from the next month's first trading
  day (`estimation_end < rebalance_date` on all 113 rebalances;
  first holding date 2017-04-03). No future data enters optimization;
  enforced by unit tests, re-verified here from `rebalance_weights.csv`.
- **Principle upheld:** every dated artifact was checked to consume
  only information available at its timestamp, with the single
  documented exception class below (research-only test predictions
  must not be relabeled as live).

## Forecast compatibility (Step 3)

`predictions.csv` / `dl_predictions.csv` provide `(date, ticker,
model, predicted_return)` with log-return horizon t → t+1, matching
the 4E forecast semantics after adapter mapping
(`date→as_of_date`, `predicted_return→predicted_log_return`) plus a
price join (`raw/market adj_close`). **They still cannot feed live
recommendations**: (a) they cover only the research test periods;
(b) no leakage-free live model-selection rule exists —
`best_model_by_ticker.csv` is test-selected (Task 2D summary states
test-RMSE selection; 5/50 non-baseline RMSE wins, mean directional
accuracy 0.4939 ≈ coin flip). Recorded verdict: **forecast provider
is currently an explicit caller/configuration dependency**
(`src/integration.py::require_explicit_model_selection` enforces it).

## Risk compatibility (Step 4)

No per-ticker current-volatility artifact exists. Backtest
`strategy_metrics.csv` holds portfolio-level (not per-ticker) research
metrics. Accepted boundary: **Risk Calculator → Recommendation
Engine**, where the calculator reads only the trailing volatility
columns of the `as_of_date` features row. Implemented as validation
contract only in this task (no new research numbers fabricated).

## Sentiment (Step 5)

Task 3C audit: all sources `suitable_for_leakage_free_3C = False`
(recent-only Yahoo, GDELT keyword/crawl-time flaws, key-gated
NewsAPI). Integration fabricates nothing: real-input validation
requires `sentiment_available=false / score=null /
status="unavailable"`; the 4E demo stays `DEMO_*`/`synthetic_demo`.

## Portfolio integration (Step 6)

Re-verified from outputs without touching the backtest: lookback 252,
strict estimation-before-rebalance, monthly first-trading-day
holding, log→simple conversion, GSPC benchmark-only, all three
strategies present, 0 failed rebalances. Dashboard-suitable (as
research exhibits, not live expectations): `strategy_metrics.csv`,
`daily_portfolio_values.csv`, `daily_portfolio_returns.csv`,
`rebalance_weights.csv`, `rebalance_turnover.csv`.

## Return conventions (Step 7)

| Component | Input | Internal | Output |
|---|---|---|---|
| Task 2B features | adj_close prices | log differences | log returns + log target |
| Task 2C classical ML | log-return features | log scale | predicted log return (t→t+1) |
| Task 3B DL | log-return sequences (30d) | log scale | predicted log return (t→t+1) |
| Task 4A math | any consistent per-period returns | same units | same units (ann. helpers explicit) |
| Task 4C frontier | log returns (demo) | sample mean/cov | return-space geometry (not a backtest) |
| Task 4D backtest | log returns | **simple** via exp−1 | simple returns, compounded values |
| Task 4E recommendation | log forecast + price | exp conversion | simple return + predicted price |

No uniformity forced: the 4D/4E log→simple conversions are intentional
and documented (exact aggregation / price forecasting respectively).

## Classification (Step 8)

Per schema §8: code modules are live-capable (A); all market-derived
artifacts are research-only (B); the 4E demo is synthetic (C);
sentiment/live feeds/live forecast provider are unavailable (D).
Final-test predictions and backtest metrics must never be displayed
as live forecasts or expected performance.

## Blockers for a LIVE pipeline

1. **No live forecast provider** — no selection rule, test predictions
   are research-only, retraining out of scope. Explicit dependency.
2. **No live risk feed** — features frozen at 2026-08-28; trailing-vol
   reader boundary defined, updater not built.
3. **No live price feed** — `current_price` has no current source.
4. Thresholds remain non-validated defaults by design.

A clean live end-to-end recommendation pipeline **cannot currently be
constructed**. This is the correct output of the audit, not a failure:
the contract module (`src/integration.py`) pins the boundaries so a
future dashboard cannot silently cross them.
