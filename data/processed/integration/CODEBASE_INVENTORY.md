# Codebase Inventory (Task 5C, Phase 1)

Baseline suite: 296 passed. Measurements: total lines / code lines
(non-blank, non-comment) / top-level functions / classes via AST.

## Python source files

| File | Lines | Code | Fn | Cls | External deps | CLI entry | Imported by | Tests |
|---|---|---|---|---|---|---|---|---|
| `src/__init__.py` | 0 | 0 | 0 | 0 | — | — | package anchor | — |
| `src/ingest.py` | 344 | 281 | 10 | 0 | pandas, yaml, yfinance | `main`, `print_availability_summary` | tests | test_ingest (13) |
| `src/features.py` | 434 | 346 | 14 | 0 | numpy, pandas, yaml | `main`, `run_features` | models, deep_learning, tests | test_features (15) |
| `src/eda.py` | 579 | 464 | 28 | 0 | numpy, pandas, yaml, matplotlib, scipy, statsmodels | `main`, `run_eda` | tests | test_eda (11) |
| `src/models.py` | 378 | 303 | 14 | 0 | numpy, pandas, yaml, sklearn, xgboost | `main`, `run_models`, `run_ticker` | evaluate, deep_learning, tests | test_models (20) |
| `src/evaluate.py` | 343 | 281 | 11 | 0 | numpy*, pandas, yaml, matplotlib | `main`, `run_evaluation` | tests | test_evaluate (15) |
| `src/deep_learning.py` | 1217 | 1027 | 54 | 12 | numpy, pandas, yaml, torch | `main`, `run_training` | tests | test_deep_learning (23), test_deep_learning_training (16) |
| `src/portfolio_math.py` | 562 | 447 | 27 | 0 | numpy, pandas | none (library) | portfolio_frontier, tests | test_portfolio_math (46) |
| `src/portfolio_optimizer.py` | 536 | 447 | 17 | 2 | numpy, pandas, scipy | none (library) | portfolio_frontier, tests | test_portfolio_optimizer (35) |
| `src/portfolio_frontier.py` | 612 | 520 | 18 | 2 | numpy, pandas, yaml, scipy | `main`, `run_frontier_analysis` | tests | test_portfolio_frontier (26) |
| `src/portfolio_backtest.py` | 699 | 588 | 8 | 1 | numpy, pandas, argparse | `main`, `run_backtest_cli` | tests | test_portfolio_backtest (17) |
| `src/recommendation.py` | 626 | 533 | 18 | 3 | pandas | `main`, `run_demo` | integration (lazy), tests | test_recommendation (30) |
| `src/integration.py` | 371 | 310 | 16 | 3 | pandas | none (contract lib) | dashboard_data (lazy), tests | test_integration (23) |
| `src/dashboard_data.py` | 407 | 307 | 22 | 0 | numpy, pandas | none (helper lib) | app, tests | test_dashboard (12), test_dashboard_integrity (15) |
| `app.py` | 557 | 469 | 21 | 0 | pandas, plotly, streamlit | `main` (streamlit run) | — | test_dashboard, test_dashboard_integrity |
| **Production total** | **7665** | **6323** | **278** | **23** | — | — | — | — |

\* `evaluate.py` imports numpy but never uses it (Phase 2 finding).

Test suite total: 16 files, 4266 lines (3132 code), ~325 test functions.
Note: test_dashboard* reference `src.models`/`src.deep_learning`/`src.features`
only as strings asserting the dashboard does NOT import them.

## Configuration / docs / entry points

- `config.yaml`: seed, universe (10 tickers, benchmark, VIX, dates, cutoff),
  data_sources, recommendation thresholds. Dashboard reads cutoff/tickers via
  `dashboard_data` constants mirroring it.
- `requirements.txt`: pandas, numpy, yfinance, pyyaml, pyarrow, pytest,
  streamlit, plotly (see Phase 6: sklearn/xgboost/torch/scipy/statsmodels/
  matplotlib missing from the file though imported).
- `README.md`, `.env.example`, methodology docs under
  `data/processed/{recommendations,portfolio_backtest,integration,
  news_availability}/`.

## Generated artifacts (KEEP — required by dashboard/tests)

- `data/raw/market/*.parquet` (13 files; gitignored but present; dashboard reads them).
- `data/processed/{features,models,evaluation,deep_learning,
  deep_learning_training/checkpoints,portfolio_frontier,portfolio_backtest,
  recommendations,integration,news_availability}/` — all consumed by dashboard
  pages or tests (see DASHBOARD_DATA_LINEAGE.md).

## Temporary / cache (DELETE-safe locally, all gitignored — left in place)

- `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/`, `.pytest_cache/`.
- No `.tmp/.log/.bak/.orig`, debug dumps, stale reports, or duplicate modules
  found. No obsolete files found: every `src/*.py` is imported by production
  code, a CLI path, or tests.
