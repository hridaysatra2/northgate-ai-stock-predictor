# Recommendation Methodology (Task 4E)

## Purpose

The recommendation engine (`src/recommendation.py`) is a transparent,
deterministic **decision-support** tool. It converts an explicitly
supplied forecast plus currently available risk information into
`BUY / HOLD / SELL` with a human-readable reason. It guarantees no
returns and predicts no market with certainty.

## No model selection inside the engine

The engine selects **no model**. Every forecast must be supplied by the
caller/configuration as `(ticker, as_of_date, current_price,
predicted_log_return, model_name)`; `model_name` is an opaque label
preserved exactly.

Why: `data/processed/evaluation/best_model_by_ticker.csv` is derived
from final test-set metrics (`evaluation_metrics.csv`:
`metric_source=predictions`, `n_test=524`), so using it for live model
choice would be test-set leakage. `selected_hyperparameters.csv` holds
dev-set CV diagnostics but the repository defines no live selection
rule on them, and inventing one here would be post-hoc methodology.
This ambiguity is resolved by not selecting at all: the caller owns
model choice and timestamp safety of the forecast.

## Forecast conversion

For log-return forecast `r_hat`:

- `predicted_simple_return = exp(r_hat) - 1`
- `predicted_price = current_price * exp(r_hat)`

Example: price 200, `r_hat = 0.02` → simple return ≈ 2.0201%,
predicted price ≈ 204.04. Log and simple returns are never confused.

## Risk classification

Only the supplied **current** annualized volatilities are used. For
each batch timestamp, `percentile(v)` = fraction of universe
volatilities at or below `v` (counting definition; ties share a value;
single-asset universe yields 1.0). Risk is `elevated` iff percentile
>= threshold (default 0.75), else `normal`. Volatility is the primary
(and only classifying) variable; Sharpe, Sortino, drawdown, and VaR
are passed through for display only (null when unavailable, never
invented), and no weighted score is constructed.

## Thresholds

Defaults from `config.yaml` (`recommendation:` section):
`positive_return_threshold: 0.01`, `negative_return_threshold: -0.01`,
`elevated_risk_percentile: 0.75`. These are **configurable defaults,
not empirically validated constants**; they were not tuned against
test results and must not be.

## Recommendation rules (exact, no scores)

- Strong positive (simple >= +thr) + normal risk → **BUY**
- Strong positive (simple >= +thr) + elevated risk → **HOLD**
- Neutral (neg_thr < simple < pos_thr) → **HOLD**
- Negative (simple <= neg_thr) + normal risk → **HOLD**
- Negative (simple <= neg_thr) + elevated risk → **SELL**

Each output carries an explicit reason (e.g. *"Forecast return is
strongly positive and current volatility is within the normal risk
range."*), plus a sentiment sentence (below). No promotional language.

## Sentiment: unavailable (Task 3C)

The Task 3C audit found no suitable historical news source, so
`sentiment_available = false`, `sentiment_score = null`,
`sentiment_status = "unavailable"`. No sentiment is invented,
scraped, substituted (GDELT/yfinance-news), or neutral-filled, and a
stock is never penalized or rewarded for missing sentiment. Every
reason states: *"Historical sentiment unavailable; recommendation uses
forecast and risk only."*

## Timestamp / leakage rules

- The engine is a pure transformation of supplied inputs: it reads no
  model predictions, evaluation metrics, market files, or outcomes
  (enforced by an automated audit test).
- Risk inputs must reflect information available at `as_of_date`. A
  historical recommendation built on full-period (2016–2026) risk
  statistics and presented as known at that date would be look-ahead
  bias; for current/latest use, trailing risk metrics are acceptable.
- No historical recommendation performance is claimed anywhere.

## Limitations

- Rules are deliberately coarse step functions with inclusive
  boundaries; small forecast changes near a threshold flip the label.
- Percentiles depend on the supplied universe composition.
- Thresholds are defaults, not optima; the engine has no notion of
  costs, liquidity, concentration, or regime.
- Outputs are decision support, not investment advice and not
  guaranteed predictions.

## Demo outputs

`demo_recommendations_SYNTHETIC.csv` (if present) uses fabricated
`DEMO_*` tickers/prices and `model_name = "synthetic_demo"` to exercise
all five rule branches. It is a **software demo, not investment
performance**, and must never be presented otherwise.
