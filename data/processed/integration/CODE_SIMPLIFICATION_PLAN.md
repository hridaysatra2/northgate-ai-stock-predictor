# Code Simplification Plan (Task 5C, Phase 11)

Rule: only LOW-risk items are applied. MEDIUM/HIGH are reported, not performed.

## LOW — applied

| # | File | Problem | Simplification | Lines removed | Why safe | Tests | Risk |
|---|---|---|---|---|---|---|---|
| L1 | `app.py` | 3 unused imports: `LIVE_RECOMMENDATIONS_EXPLANATION`, `is_synthetic_ticker`, `try_generate_live_recommendation` (import line is the only occurrence) | Remove from import list | 3 | No other reference in file; none in `__all__` or required by tests | test_dashboard (12), integrity (15) | LOW |
| L2 | `src/evaluate.py` | `import numpy as np` with zero uses | Remove line | 1 | AST + text search confirm unused | test_evaluate (15) | LOW |
| L3 | `src/recommendation.py` | `field` imported from dataclasses, never used | `from dataclasses import dataclass, field` → `dataclass` | 0 (edit) | Text search: single occurrence | test_recommendation (30) | LOW |
| L4 | `src/dashboard_data.py` | `SYNTHETIC_TICKER_PREFIX` / `SYNTHETIC_MODEL_NAME` redefined; canonical copies live in `src/integration.py` with identical values | Import from `src.integration` instead of redefining (keeps `dd.*` attribute access) | 2 | Values identical (`DEMO_`/`synthetic_demo`); no import cycle (integration imports only math/dataclasses/pandas at top level) | test_dashboard (5,6), integrity (6), test_integration | LOW |

## MEDIUM — reported, NOT performed

- **M1. Six `load_config` variants** (`ingest/features/eda/models/evaluate/deep_learning`): genuine
  duplication, but each carries module-specific defaults/validation wired into
  protected research flows. Centralizing touches 6 research modules.
- **M2. `requirements.txt` missing direct deps**: `scikit-learn`, `xgboost`,
  `torch`, `scipy`, `statsmodels`, `matplotlib` are imported by production code
  but not declared; `streamlit`/`plotly` unpinned. Fresh-install correctness
  issue, but pinning unknown-good versions risks breaking the working env.
- **M3. `deep_learning.py` local aliases** (`_json`/`_pd`/`_sys`/`_random`,
  each used once): unifying to top-level imports saves ~4 lines inside protected
  training code; benefit does not justify churn.

## HIGH — reported, NOT performed

- **H1. Long research functions** (`run_rolling_backtest` 235 lines,
  `run_eda` 150, `prepare_ticker` 108, `run_training` 105, `efficient_frontier`
  91): each is a single validated research procedure; splitting risks
  methodology drift. `render_stock_analysis` (86) is one cohesive page.
- **H2. Seed/config wiring** (`random_seed`, `data_sources.market` unread from
  config; seeds hardcoded as 42): changing to config-driven values would alter
  reproducibility guarantees and test expectations. Documented, not changed.
- **H3. Test suite**: no deletions. Leakage/timestamp/isolation/safety tests are
  intentionally repetitive; 296 tests all assert meaningful behavior.

## Deliberately retained (look redundant, are required)

- `print()` in all `run_*`/`main` CLI blocks: user-facing CLI output, not debug.
- `from src.models import compute_metrics` (deep_learning, function-level):
  avoids a circular import; keep lazy.
- Verbose validators in `integration.py`/`recommendation.py`/`portfolio_*`:
  leakage/timestamp/safety guards — the audit's Phase 10 protections.
- Full-precision display tables in dashboard: research-terminal choice.
- `__pycache__`/`.pytest_cache`: gitignored regenerable caches; left in place.
