# Northgate AI Stock Predictor

Quantitative research and decision-support pipeline: market-data ingestion
(yfinance) → EDA → supervised features (`target_t = log(AdjClose_{t+1} /
AdjClose_t)`) → classical ML (chronological split, TimeSeriesSplit CV) →
evaluation → deep-learning sequences/training → portfolio math, optimization,
rolling backtest → rule-based recommendations/integration → Streamlit
dashboard. No live trading claims.

## Purpose and scope

Decision-support research only: data-quality checks, leakage-free features,
chronological validation, out-of-sample evaluation, portfolio illustration,
and explicitly labeled synthetic demos. Live recommendation generation is
not implemented; sentiment analysis is absent (see limitations).

## Prerequisites and installation

- Python 3.10, `pip install -r requirements.txt`.
- CUDA PyTorch is pinned as `torch==2.8.0+cu126` (local CUDA build).
  Fresh CUDA installs need the PyTorch CUDA index, e.g.
  `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`.
- CPU-only alternative: `pip install torch==2.8.0` (same code path;
  `src/deep_learning.py` uses CUDA when available, else CPU). Training is
  slower on CPU but preparation, tests, and the dashboard do not need a GPU.

## Configuration and cutoff

`config.yaml`: `random_seed: 42`, US market, 10 equities
(AAPL/MSFT/JPM/JNJ/PG/XOM/HD/V/CAT/KO), benchmark `^GSPC`, risk ticker
`^VIX`, `start_date: 2016-01-01`, `research_cutoff_date: 2026-08-31`.
Filenames strip `^` (e.g. `GSPC.parquet`). All research views enforce
`date <= 2026-08-31`; backtest `--end` beyond the cutoff raises loudly.

## How to run

- Ingestion: `python -m src.ingest` (downloads via yfinance with
  `auto_adjust=False`, writes `data/raw/market/*.parquet`).
- EDA: `python -m src.eda` (reads raw only; writes
  `data/processed/eda/{tables,plots}/`).
- Features: `python -m src.features` (writes
  `data/processed/features/*.parquet` + `feature_metadata.csv`).
- Classical models: `python -m src.models` (chronological 80/20 + 1-row
  embargo, `TimeSeriesSplit(5, gap=1)` on dev only; writes
  `data/processed/models/`).
- Evaluation: `python -m src.evaluate` (reads models only; writes
  `data/processed/evaluation/`, including `best_model_by_ticker.csv`).
- Deep learning: `python -m src.deep_learning --prepare` (writes
  `data/processed/deep_learning/`), then `python -m src.deep_learning`
  (trains LSTM/GRU/BiLSTM/Transformer; writes
  `data/processed/deep_learning_training/` + `checkpoints/`).
- Portfolio illustration: `python -m src.portfolio_frontier`
  (writes `data/processed/portfolio_frontier/`; in-sample demo, not a
  backtest).
- Backtest: `python -m src.portfolio_backtest`
  (`--output-dir/--start/--end/--lookback/--initial-capital/--risk-free-rate-annual/--sortino-target/--var-confidence/--tickers`;
  writes `data/processed/portfolio_backtest/`).
- Recommendations demo: `python -m src.recommendation` (synthetic only;
  writes `data/processed/recommendations/demo_recommendations_SYNTHETIC.csv`).
- Dashboard: `streamlit run app.py` (read-only; missing artifacts show
  `Unavailable`, never synthesized).

## Source modules

- `src/ingest.py`: fetch/normalize/cache; cutoff, sort/dedup, gap reports.
- `src/eda.py`: descriptive stats/plots; trailing windows, no fill,
  date-indexed correlation.
- `src/features.py`: 30 leakage-free features + 1-day-ahead log-return
  target; trailing/lag/calendar/market handling.
- `src/models.py` / `src/evaluate.py`: classical ML + reporting-only
  evaluation; zero random-walk baseline.
- `src/deep_learning.py`: 30-day sequences (70/15/15 + 1-sequence embargo),
  train-only scaler, LSTM/GRU/BiLSTM/Transformer with val-only early
  stopping and single test evaluation.
- `src/portfolio_math.py` / `src/portfolio_optimizer.py` /
  `src/portfolio_frontier.py` / `src/portfolio_backtest.py`: return
  math, long-only SLSQP, Monte Carlo/frontier illustration, monthly
  rolling backtest.
- `src/recommendation.py` / `src/integration.py`: 5-branch forecast rules
  (`expm1` log→simple), strict validation contract, test-selected files
  can never become live forecasts.
- `src/dashboard_data.py` / `app.py`: read-only Streamlit data layer and
  pages with cutoff clipping, unit labels, and synthetic-vs-real badges.

## Tests

Run targeted or full suites offline (no network; yfinance stubbed):

- `python -m pytest tests/test_eda_alignment.py tests/test_eda.py -v`
- `python -m pytest tests/test_dl_embargo.py tests/test_deep_learning.py -v`
- `python -m pytest tests -v`

## Key findings (current regenerated results)

Classical models did not consistently outperform the zero-return
random-walk baseline on RMSE; directional accuracy is near coin-flip
(~0.49-0.51). Current `data/processed/deep_learning_training/dl_metrics.csv`:
mean DL RMSE 0.0174 across 44 ticker/model rows; see
`data/processed/evaluation/evaluation_summary.txt` for the classical
baseline finding and DL `config.json` for run metadata.
`best_model_by_ticker.csv` selects winners on test
metrics for reporting only and must not drive live selection.

## Limitations

- No suitable historical news feed was available; sentiment is absent
  (`sentiment_available=False`, score null; see
  `data/processed/news_availability/`). Nothing is inferred in its place.
- No complete live recommendation data pipeline exists
  (`try_generate_live_recommendation` always raises
  `MissingDependencyError`); the dashboard shows an unavailable banner.
- Backtest is gross of transaction costs (excluded, turnover reported),
  long-only fully-invested, monthly rebalance on the first trading day
  with a 252-day lookback strictly before each decision; failed
  optimizations hold prior weights or cash transparently.
- Recommendation demo is synthetic only (`DEMO_*` tickers,
  `model_name="synthetic_demo"`, `as_of_date=2026-08-28`,
  `*_SYNTHETIC.csv`); zero real-ticker rows.

## Outputs and missing artifacts

Present under `data/processed/`: `features/`, `models/`, `evaluation/`,
`deep_learning/`, `deep_learning_training/`, `portfolio_frontier/`,
`portfolio_backtest/`, `recommendations/`, `news_availability/`,
`integration/`. `data/processed/eda/` is absent in this checkout
(dashboard reports EDA sections as unavailable). `data/raw/` is local-only
(git-ignored). The `data/processed/deep_learning/` datasets use the corrected
one-sequence embargo, and the latest training run completed 44/44 model runs
with zero reported failures (seed 42, PyTorch 2.8.0+cu126, 19,832 training
sequences); CUDA determinism is best-effort, not strict bit-for-bit
reproducibility, and results carry no profitability claim.

## Task 3A — Leakage-free deep-learning sequence preparation

Preparation ONLY (no LSTM/GRU/BiLSTM/Transformer training or tuning here).

- **Lookback:** 30 trading days. `X[i]` = observations t-29..t, `y[i]` = `target_t`;
  the final step of `X[i]` is exactly timestamp t.
- **Target:** `target_t = log(AdjClose_{t+1} / AdjClose_t)`; every input feature
  for t is available no later than t. Incomplete windows and invalid targets are
  removed; nothing is filled, interpolated, or synthesized.
- **Split:** chronological 70% / 15% / 15% on prediction timestamps (target rows),
  never shuffled, plus a 1-sequence embargo at train/val and val/test so the
  target on the last train (val) row never ends on the first val (test) price.
  Windows slide over the full ordered history and are assigned
  by endpoint t, so val/test windows may include pre-boundary history but no
  val/test TARGET ever enters training and no observation later than t enters
  the window predicting t.
- **Scaling:** `TrainOnlyStandardScaler` (mean/std) fitted on training rows only;
  target never scaled; train/val/test transformed with the frozen training
  statistics. Zero-variance features get scale 1.0.
- **Leakage rules tested:** exact endpoint→target mapping, no t+1 in window,
  no future row in earlier windows, disjoint ordered splits
  (train < val < test), boundary-target embargo, train-only scaler stats,
  test-mutation invariance, determinism, raw/feature immutability, loud failures
  (missing ticker/file, VIX-without-target, missing columns, too few rows,
  non-finite features).
- **Outputs:** `data/processed/deep_learning/` — `<TICKER>.npz` (X/y/dates per
  split + feature names), `<TICKER>_metadata.json` (ticker, lookback, feature
  count/names, row + sequence counts, first/last prediction timestamp per split,
  scaler mean/scale), `dataset_summary.csv`. No duplicate raw data is stored.
- **Run:** `python -m src.deep_learning --prepare` then
  `python -m pytest tests/test_deep_learning.py -v` and `python -m pytest tests -v`.
  Supervised tickers: 10 equities + GSPC (VIX has no target and raises loudly).

## Task 3B — Leakage-free deep-learning training (LSTM/GRU/BiLSTM/Transformer)

Training + evaluation ONLY on the frozen Task 3A datasets
(`data/processed/deep_learning/`). Raw data, `data/processed/features/`,
and Task 2C/2D outputs are read-only and untouched.

- **Target:** `target_t = log(AdjClose_{t+1} / AdjClose_t)`; never raw price.
- **Input shape:** 30 trading days x 30 features; **output:** one scalar
  predicted next-day log return per sequence.
- **Models (modest, shared config):** LSTM (hidden 64, 1 layer, dropout 0.1),
  GRU (same), BiLSTM (bidirectional, 2x64 head; bidirectionality runs only
  within the historical 30-step window t-29..t, never past t), Transformer
  (encoder-only: 30->d_model 64 projection, sinusoidal positions, 2 layers,
  4 heads, dim_feedforward 128, dropout 0.1; last-step head -> scalar).
- **Optimizer/loss:** Adam (lr 1e-3, weight_decay 1e-5), MSELoss, batch 64,
  max 50 epochs.
- **Early stopping:** validation loss ONLY (patience 8); best-val checkpoint
  restored. No train+val refit.
- **Validation-only model selection; test isolation:** architecture, config,
  and checkpoint are fixed before the test set is touched; each finalized
  model is evaluated exactly once on TEST. Test is never used for tuning,
  early stopping, or architecture choice.
- **Metrics:** RMSE, MAE (primary), MAPE (zero-safe, unstable near zero),
  R2, directional accuracy — identical definitions to Task 2C
  (`src.models.compute_metrics`); direction reported separately, no
  profitability claims.
- **Reproducibility:** seed 42 (Python/NumPy/PyTorch; cuDNN deterministic,
  `use_deterministic_algorithms(warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG`
  recommended for stricter CUDA determinism). Device: CUDA (NVIDIA GeForce
  RTX 4060 Laptop GPU, 8 GB) when available, else CPU. Seed/device/torch
  version recorded in `config.json`.
- **Outputs:** `data/processed/deep_learning_training/` — `dl_metrics.csv`
  (ticker/model/RMSE/MAE/MAPE/R2/directional_accuracy/best_epoch/
  best_validation_loss/seed), `training_history.csv`
  (ticker/model/epoch/train_loss/validation_loss), `dl_predictions.csv`,
  `dl_best_models.csv`, `config.json`, `checkpoints/<TICKER>_<model>_best.pt`.
- **Command used to train:** `python -m src.deep_learning`
  (Task 3A preparation via `python -m src.deep_learning --prepare`).
- **Result (honest, current run):** 44/44 runs succeeded with zero reported
  failures; mean DL RMSE 0.0174 across 44 ticker/model rows with mean
  directional accuracy ~0.49 (coin flip) — consistent with the classical-model
  finding of no demonstrated superiority over the baseline.
  No predictive-signal or profitability claim.
