# Dashboard Data Lineage (Task 5B)

Every research number shown in `app.py` traces to a frozen artifact below.
The dashboard is read-only: it never writes artifacts, never retrains, never
optimizes. Display-only derivations are marked **[D]** with their exact formula.
All research views are clipped to `RESEARCH_CUTOFF = 2026-08-31`
(`src.dashboard_data.clip_to_cutoff`); raw files currently end 2026-08-31,
features/frontier/backtest end 2026-08-28 (target shift / weekends).

Conventions used throughout (Tasks 4A/4B/4C/4D):
- **log return**: `log(adj_t / adj_{t-1})`, per-day, unitless.
- **simple return**: `exp(log) - 1`, per-day.
- **annualized volatility**: daily sample std × √252.
- **predicted_log_return / predicted_simple_return**: forecast over (t, t+1],
  never realized return. Conversion: `simple = exp(log) - 1`,
  `predicted_price = current_price × exp(log)`.
- **directional accuracy**: proportion in [0, 1]; 0.509 = 50.9%, 0.50 ≈ coin flip.

## 1. Overview

| Dashboard element | Source artifact | Source column / field | Transformation | Displayed unit |
|---|---|---|---|---|
| Research cutoff KPI | `config.yaml` (`universe.research_cutoff_date`) via `RESEARCH_CUTOFF` | — | none (constant) | ISO date |
| Equities KPI (10) | `config.yaml` universe tickers via `TICKERS` | — | `len()` | count |
| Benchmark KPI | `BENCHMARK_LABEL` | — | none | "S&P 500" |
| Risk regime KPI | `RISK_REGIME_TICKER` | — | none | "VIX" |
| Data period KPI | `RESEARCH_START` → `RESEARCH_CUTOFF` | — | none | ISO dates |
| Total feature rows | `data/processed/deep_learning/dataset_summary.csv` | `n_rows_total` summed over 11 rows (10 equities + GSPC) | **[D]** `sum()` | count of rows |
| Universe line | `TICKERS` + `BENCHMARK_TICKER` | — | join | tickers |
| Findings code block | `data/processed/evaluation/evaluation_summary.txt` | full text | verbatim | text |
| Aggregate table | `data/processed/evaluation/aggregate_model_summary.csv` | all columns (`mean_rmse`, `mean_directional_accuracy`, …) | none (copy) | log-return units; proportion |
| Artifact availability | filesystem probe per `ARTIFACT_PATHS` | existence | Available/Unavailable | status |

## 2. Stock Analysis (per selected ticker)

| Dashboard element | Source artifact | Source column | Transformation | Displayed unit |
|---|---|---|---|---|
| KPI: observations | `data/raw/market/<T>.parquet` | row count (≤ cutoff) | **[D]** `len()` | count |
| KPI: last adj close | same | `adj_close` last row | none | USD price |
| KPI: skewness / kurtosis | same | `adj_close` → **[D]** daily log returns `log(p_t/p_{t-1})`, then `skew()`/`kurt()` (pandas, sample) | **[D]** as stated | unitless |
| Period caption | same | `date` min/max | none | ISO dates |
| Adj-close + SMA chart | same | `date`, `adj_close`; **[D]** trailing `rolling(20/50).mean()` | **[D]** trailing means | USD |
| Normalized vs S&P 500 | `data/raw/market/<T>.parquet` + `data/raw/market/GSPC.parquet` | `adj_close` inner-joined on date | **[D]** `price/first×100` per series | rebased index (100 = first) |
| Rolling volatility 20d | `data/processed/features/<T>.parquet` | `log_return_1d` → **[D]** `rolling(20).std(ddof=1)×√252` | **[D]** as stated | annualized vol (fraction) |
| Return histogram | same | `log_return_1d` | none (binned) | daily log return |
| Drawdown curve | raw `adj_close` | **[D]** `price/cummax-1` (`compute_drawdown`) | **[D]** as stated | negative fraction |
| ADF section | `data/processed/eda/tables/adf_results.csv` | statistic, p-value, lags | none — **currently absent**: shows documented reason, no recompute | — |

## 3. Model Performance (Tasks 2D + 3B, historical test evaluation only)

| Dashboard element | Source artifact | Source column | Transformation | Displayed unit |
|---|---|---|---|---|
| Mean test RMSE bar | `data/processed/evaluation/aggregate_model_summary.csv` | `mean_rmse` | none; y-axis `rangemode=tozero` | log-return units |
| Mean directional accuracy bar | same | `mean_directional_accuracy` | none; y-axis fixed [0, 1] | proportion (0.50 = coin flip) |
| Aggregate table | same | all columns | none | as above; R² variance-explained; MAPE % (unstable near zero) |
| Per-ticker RMSE bar + full table | `data/processed/evaluation/evaluation_metrics.csv` | `rmse` filtered by model | none; y-axis `tozero` | log-return units |
| Baseline comparison table | `data/processed/evaluation/baseline_comparison.csv` | `rmse`, `baseline_rmse`, `rmse_improvement` | none | log-return units; improvement fraction |
| Best test-set RMSE table | `data/processed/evaluation/best_model_by_ticker.csv` | `best_model`, `rmse`, … | none; labeled "Best test-set RMSE — historical evaluation only"; warning that test selection must not drive live model choice | log-return units |
| DL table + chart | `data/processed/deep_learning_training/dl_metrics.csv` | `ticker`, `model`, `rmse`, … | none | log-return units; proportion |
| Random Walk baseline | present in all classical tables (`random_walk` rows; zero-return forecast) | — | none | — |

## 4. Portfolio Optimization (Task 4C, in-sample illustration)

| Dashboard element | Source artifact | Source column | Transformation | Displayed unit |
|---|---|---|---|---|
| MC cloud + frontier + key portfolios | `monte_carlo_portfolios.csv` / `efficient_frontier.csv` / `key_portfolios.csv` | `volatility` (x), `expected_return` (y) | none | **daily** log-return units (per `frontier_metadata.json: return_definition`) |
| Metadata (n_obs, dates, estimator, rf) | `frontier_metadata.json` | `n_obs=2619`, `date_min/max`, `estimator=sample mean/covariance (ddof=1)`, `risk_free_rate=0.0` | verbatim subset | as labeled |
| Key-portfolio table | `key_portfolios.csv` | `expected_return`, `variance`, `volatility`, `sharpe_ratio`, 10 weight columns | none | daily units; weights sum to 1 |
| Weight inspector | same | selected row's 10 ticker columns | none | weight fraction |
| Benchmark note | — | — | UI states GSPC is evaluation-only, never a constituent (verified: weight assets = 10 equities only) | — |

## 5. Portfolio Backtest (Task 4D — Historical rolling backtest)

| Dashboard element | Source artifact | Source column | Transformation | Displayed unit |
|---|---|---|---|---|
| Cumulative value lines | `daily_portfolio_values.csv` (clipped ≤ cutoff) | `date`, `equal_weight`, `rolling_min_variance`, `rolling_max_sharpe`, `sp500` | melt to long; none | USD portfolio value (from 100000.0) |
| Drawdown illustration | same (`equal_weight`) | **[D]** `value/cummax-1` | **[D]** as stated | negative fraction |
| Summary JSON | `backtest_summary.json` | `actual_first_date=2017-04-03`, `actual_last_date=2026-08-28`, `n_rebalance_dates=113`, `costs=gross, NOT modeled`, `disclaimer` | verbatim subset | as labeled |
| Strategy metrics table | `strategy_metrics.csv` | `cumulative_return` (multiple of initial capital, e.g. 4.25), `annualized_return_geo` (geometric annual rate), `annualized_volatility` (×√252), `sharpe_ratio`/`sortino_ratio` (ratios), `maximum_drawdown` (negative fraction), `calmar_ratio`, `historical_var_daily` (positive daily loss, 95%), `total_turnover` (Σ\|Δw\|), `n_observations=2364` | none | per-column units in UI caption |
| Method assumptions | `BACKTEST_METHODOLOGY.md` + `BACKTEST_ASSUMPTIONS` | 252-day window strictly before each month-end rebalance; weights held [rebalance, next); long-only; no costs | quoted | — |

## 6. Recommendations (Task 4E engine implemented; production unavailable per 4F)

| Dashboard element | Source artifact | Transformation |
|---|---|---|
| "Live recommendations unavailable" + explanation | `get_live_recommendation_status()` (constants: no forecast/price/risk feed, no sentiment) | none |
| Real-ticker row count (0) | `demo_recommendations_SYNTHETIC.csv` → `filter_real_recommendations()` drops all `DEMO_*`/`synthetic_demo` rows | **[D]** filter only |
| Demo table | same file; `label_demo_frame()` adds `demo_label = "SYNTHETIC DEMO — NOT A REAL RECOMMENDATION"` | **[D]** label column only |
| Per-row badges | `is_synthetic_record()` per row | none |
| `best_model_by_ticker.csv` | **never loaded on this page**; test-selected file cannot become live model selection (`require_explicit_model_selection` raises without caller-supplied model) | n/a |

Demo file semantics (both return conventions shown side by side):
`predicted_log_return` (log) → `predicted_simple_return = exp(log)-1` → `predicted_price = current_price×exp(log)`.

## 7. Methodology & Limitations

Static text using exact project terminology (yfinance 2016-01-01→2026-08-31,
adjusted close, leakage-safe trailing features, 1-day-ahead log-return target,
chronological/TimeSeriesSplit/gap-embargo/train-only scaling, 10 named models +
Random Walk, MPT long-only 252-day monthly, sentiment unavailable Task 3C,
rule-based configurable-threshold risk-aware recommendations with no live
provider) plus verbatim excerpts (`NEWS_DATA_AVAILABILITY.md`,
`RECOMMENDATION_METHODOLOGY.md`, `INTEGRATION_AUDIT.md` excerpt).
