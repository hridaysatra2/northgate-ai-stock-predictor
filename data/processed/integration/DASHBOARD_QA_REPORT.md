# Dashboard QA Report (Task 5B)

Audit of `app.py` + `src/dashboard_data.py` against frozen research artifacts.
Baseline suite 282 passed; no research methodology changed; dashboard/tests only.

## 1. Issues discovered (and disposition)

1. **No cutoff enforcement in loaders (latent).** Raw files currently end
   exactly 2026-08-31, so nothing leaked — but a future raw row would have
   entered research views. **Fixed**: `clip_to_cutoff()` in `dashboard_data.py`,
   applied in `load_raw_prices`, `load_feature_frame`, and the backtest value
   view. Covered by new cutoff tests.
2. **Directional-accuracy chart auto-zoom.** Values (~0.49–0.51) rendered with a
   truncated axis, exaggerating noise as signal. **Fixed**: y-axis fixed to
   [0, 1], title/caption state "proportion; 0.50 is coin flip", new
   `format_directional_accuracy()` helper (0.509 → "50.9%").
3. **RMSE bar charts auto-zoom.** 0.0153 vs 0.0158 gaps visually exaggerated.
   **Fixed**: `rangemode="tozero"` on RMSE charts; axis labels state
   log-return units.
4. **Backtest metrics lacked a unit legend.** `cumulative_return = 4.25` could be
   misread as annualized. **Fixed**: caption defines every column's unit
   (multiple vs geometric annual rate vs ratios vs negative-fraction drawdown
   vs positive daily-loss VaR vs turnover).
5. **ADF section hardcoded "not connected".** The real state is narrower: no
   `data/processed/eda/` outputs exist in this checkout (Task 2A module
   `src.eda.adf_summary` exists; its CSV was never generated/committed).
   **Fixed**: `load_adf_results()` probes the artifact, renders it if present,
   else shows the exact missing path + correct null-hypothesis interpretation.
   No ADF recomputation (per instruction).
6. **"Total feature rows" ambiguous** (sums 10 equities + GSPC). **Fixed**:
   relabeled "Total feature rows (10 equities + benchmark)".
7. **Benchmark-isolation stated nowhere in optimization UI** (only enforced in
   `src/integration.py`). **Fixed**: caption states GSPC is evaluation-only,
   never a constituent; new test verifies weight assets.

## 2. Audits with no change required (verified)

- **Return conventions**: log vs simple never confused; histogram/volatility use
  `log_return_1d`; demo table shows both forecast columns; no "0.02 as 2%"
  mislabel found.
- **Model metrics**: RMSE/MAE (log-return units), R² (variance-explained),
  MAPE (flagged unstable), directional accuracy (proportion, no %-point error,
  no superiority claim near 50%, no "best for trading" language, Random Walk
  present in all classical views, best_by_ticker labeled historical + leakage
  warning).
- **Stock views**: adj close labeled, SMA windows labeled, volatility labeled
  annualized, drawdown negative fraction, no forecast language.
- **Backtest**: historical labeling, 252-day/monthly/strictly-before/no-costs/
  long-only all match `BACKTEST_METHODOLOGY.md` + `backtest_summary.json`.
- **Recommendations**: live unavailable banner intact; `best_model_by_ticker.csv`
  never loaded on the page; demo rows all `DEMO_*`/`synthetic_demo` with
  prominent labels.
- **Missing artifacts**: every page shows an "Unavailable … (no data
  fabricated)" state naming the expected path.

## 3. Remaining limitations (not defects)

- EDA/ADF outputs absent (`data/processed/eda/` does not exist); section shows
  the documented Unavailable state.
- No live forecast/price/risk/sentiment feeds — dashboard remains a
  research/decision-support interface, not production trading software.
- Tables show full artifact precision (deliberate for a research terminal).
