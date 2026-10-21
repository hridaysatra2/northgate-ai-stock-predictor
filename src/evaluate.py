"""Classical-model evaluation & comparison (Task 2D).

EVALUATION ONLY: reads the already-completed Task 2C outputs under
``data/processed/models/`` and writes a reproducible comparison report to
``data/processed/evaluation/``. Never trains, tunes, or modifies features,
raw data, or Task 2C outputs.

Metric definitions are imported from ``src.models.compute_metrics`` so they
are identical to Task 2C by construction. When prediction-level rows exist,
metrics are independently recomputed from those predictions; otherwise the
stored ``model_metrics.csv`` values are used and that limitation is recorded
in ``metric_source`` and in the summary text.

Outputs (deterministic; no timestamps or random IDs inside files):
  evaluation_metrics.csv, baseline_comparison.csv, best_model_by_ticker.csv,
  aggregate_model_summary.csv, plots/*.png, evaluation_summary.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; default styles only, no seaborn
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.models import MODEL_NAMES, compute_metrics

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/processed/models")
EVAL_DIR = Path("data/processed/evaluation")
PLOTS_DIR = EVAL_DIR / "plots"

BASELINE_MODEL = "random_walk"
REQUIRED_METRIC_FIELDS = ["rmse", "mae", "mape", "r2", "directional_accuracy"]



def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load the equity universe from config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_outputs(models_dir: str | Path) -> tuple[pd.DataFrame,
                                                        pd.DataFrame | None]:
    """Load Task 2C metrics (required) and predictions (optional).

    Raises FileNotFoundError with a useful message when the required
    metrics file is absent (e.g. production training was never run).
    """
    models_dir = Path(models_dir)
    metrics_path = models_dir / "model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"required Task 2C output missing: {metrics_path}. "
            "Run the Task 2C production pipeline first; this evaluator "
            "never trains models itself and will not fabricate data.")
    metrics = pd.read_csv(metrics_path)
    preds = None
    pred_path = models_dir / "predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path, parse_dates=["date"])
    return metrics, preds


def validate_outputs(metrics: pd.DataFrame, preds: pd.DataFrame | None,
                     tickers: list[str]) -> pd.DataFrame:
    """Validate completeness/consistency; return canonical metrics frame.

    Checks: required fields, every ticker x MODEL_NAMES present,
    random_walk present per ticker, and (when predictions exist) identical
    test-date sets across models per ticker plus no duplicate
    ticker/model/date rows. Metrics are recomputed from predictions when
    available (metric_source='predictions'), else stored values are used
    (metric_source='stored').
    """
    missing = [c for c in ["ticker", "model", *REQUIRED_METRIC_FIELDS]
               if c not in metrics.columns]
    if missing:
        raise ValueError(f"model_metrics.csv missing columns: {missing}")
    expected = {(t, m) for t in tickers for m in MODEL_NAMES}
    found = set(zip(metrics["ticker"], metrics["model"]))
    if expected - found:
        raise ValueError(
            f"missing ticker/model combinations: {sorted(expected - found)}")
    for t in tickers:
        if "random_walk" not in set(metrics[metrics["ticker"] == t]["model"]):
            raise ValueError(f"random_walk baseline missing for {t}")

    if preds is None:
        out = metrics.copy()
        out["metric_source"] = "stored"
        return out

    dup = preds.duplicated(subset=["ticker", "model", "date"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate ticker/model/date predictions")
    for t in tickers:
        date_sets = {m: set(g["date"]) for m, g in
                     preds[preds["ticker"] == t].groupby("model")}
        if len({frozenset(s) for s in date_sets.values()}) != 1:
            raise ValueError(
                f"models have different test observations for {t}")
    rows = []
    for (t, m), g in preds.groupby(["ticker", "model"]):
        g = g.sort_values("date")
        rec = compute_metrics(g["actual_return"].to_numpy(float),
                              g["predicted_return"].to_numpy(float))
        rows.append({"ticker": t, "model": m, **rec,
                     "metric_source": "predictions",
                     "n_test": len(g)})
    return pd.DataFrame(rows)



def baseline_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per ticker/model RMSE improvement vs the zero-return baseline.

    rmse_improvement = (baseline_rmse - model_rmse) / baseline_rmse.
    Positive = beats baseline. R2 is NOT used as a baseline comparison.
    """
    base = metrics[metrics["model"] == BASELINE_MODEL].set_index("ticker")
    rows = []
    for _, r in metrics.iterrows():
        b = float(base.loc[r["ticker"], "rmse"])
        m = float(r["rmse"])
        rows.append({
            "ticker": r["ticker"], "model": r["model"],
            "rmse": m, "baseline_rmse": b,
            "rmse_abs_diff": b - m,
            "rmse_improvement": (b - m) / b if b != 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["ticker", "model"]).reset_index(
        drop=True)


def best_model_by_ticker(metrics: pd.DataFrame,
                         comparison: pd.DataFrame) -> pd.DataFrame:
    """Lowest-RMSE model per ticker (reporting only; no further tuning).

    WARNING: winners are selected on TEST-set metrics. This table must not
    be used as a live model-selection mechanism; live choice needs a fresh
    holdout or explicit caller configuration.
    """
    comp = comparison.set_index(["ticker", "model"])
    rows = []
    for t, g in metrics.groupby("ticker"):
        best = g.loc[g["rmse"].idxmin()]
        rows.append({
            "ticker": t, "best_model": best["model"],
            "rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"],
            "directional_accuracy": best["directional_accuracy"],
            "rmse_improvement_vs_random_walk": comp.loc[
                (t, best["model"]), "rmse_improvement"],
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def aggregate_summary(metrics: pd.DataFrame,
                      comparison: pd.DataFrame) -> pd.DataFrame:
    """Cross-ticker summary statistics per model (no pooled predictions)."""
    comp = comparison.set_index(["ticker", "model"])
    rows = []
    for m, g in metrics.groupby("model"):
        impr = comp.loc[(slice(None), m), "rmse_improvement"]
        rows.append({
            "model": m,
            "mean_rmse": g["rmse"].mean(),
            "median_rmse": g["rmse"].median(),
            "mean_mae": g["mae"].mean(),
            "median_mae": g["mae"].median(),
            "mean_r2": g["r2"].mean(),
            "mean_directional_accuracy": g["directional_accuracy"].mean(),
            "n_tickers_beat_baseline": int((impr > 0).sum()),
            "n_tickers_worse_than_baseline": int((impr < 0).sum()),
            "n_tickers": len(g),
        })
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    out = pd.DataFrame(rows)
    return out.sort_values(
        "model", key=lambda s: s.map(order)).reset_index(drop=True)



def save_fig(fig: plt.Figure, path: str | Path) -> None:
    """Save a figure with default styling and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_plots(metrics: pd.DataFrame, comparison: pd.DataFrame,
               aggregate: pd.DataFrame, plots_dir: str | Path) -> list[str]:
    """Four readable report plots; returns saved filenames."""
    plots_dir = Path(plots_dir)
    made = []
    pivot = metrics.pivot(index="ticker", columns="model",
                          values="rmse")[MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Test RMSE by ticker and model (lower is better)")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    save_fig(fig, plots_dir / "rmse_by_ticker_model.png")
    made.append("rmse_by_ticker_model.png")

    imp = comparison[comparison["model"] != BASELINE_MODEL].pivot(
        index="ticker", columns="model", values="rmse_improvement")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    imp.plot(kind="bar", ax=ax)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("RMSE improvement vs random_walk (>0 beats baseline)")
    ax.set_ylabel("Relative improvement")
    fig.tight_layout()
    save_fig(fig, plots_dir / "rmse_improvement_vs_baseline.png")
    made.append("rmse_improvement_vs_baseline.png")

    da = metrics.pivot(index="ticker", columns="model",
                       values="directional_accuracy")[MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    da.plot(kind="bar", ax=ax)
    ax.axhline(0.5, linewidth=1.0)
    ax.set_title("Directional accuracy by ticker/model (0.5 = coin flip)")
    ax.set_ylabel("Directional accuracy")
    fig.tight_layout()
    save_fig(fig, plots_dir / "directional_accuracy.png")
    made.append("directional_accuracy.png")

    agg = aggregate.set_index("model").loc[MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(agg.index.tolist(), agg["mean_rmse"].tolist())
    ax.set_title("Mean test RMSE by model (lower is better)")
    ax.set_ylabel("Mean RMSE")
    fig.tight_layout()
    save_fig(fig, plots_dir / "mean_rmse_by_model.png")
    made.append("mean_rmse_by_model.png")
    return made


def write_summary(metrics: pd.DataFrame, comparison: pd.DataFrame,
                  best: pd.DataFrame, aggregate: pd.DataFrame,
                  path: str | Path) -> None:
    """Honest narrative summary; no trading-profitability claims."""
    agg = aggregate.set_index("model")
    lowest_rmse = agg["mean_rmse"].idxmin()
    highest_da = agg["mean_directional_accuracy"].idxmax()
    most_beats = aggregate.loc[
        aggregate["n_tickers_beat_baseline"].idxmax()]
    beats = int(aggregate[aggregate["model"] != BASELINE_MODEL][
        "n_tickers_beat_baseline"].sum())
    total = int(aggregate[aggregate["model"] != BASELINE_MODEL][
        "n_tickers"].sum())
    da_overall = metrics[metrics["model"] != BASELINE_MODEL][
        "directional_accuracy"].mean()
    n_best_baseline = int((best["best_model"] == BASELINE_MODEL).sum())
    lines = [
        "Northgate AI Stock Predictor — Task 2D evaluation summary",
        "(classical ML test results; evaluated once, not used for tuning)",
        "",
        f"Tickers evaluated: {metrics['ticker'].nunique()}; "
        f"models: {', '.join(MODEL_NAMES)}.",
        f"Lowest mean test RMSE: {lowest_rmse} "
        f"({agg.loc[lowest_rmse, 'mean_rmse']:.6f}).",
        f"Highest mean directional accuracy: {highest_da} "
        f"({agg.loc[highest_da, 'mean_directional_accuracy']:.4f}).",
        f"Most baseline wins: {most_beats['model']} beats random_walk on "
        f"{int(most_beats['n_tickers_beat_baseline'])} tickers by RMSE.",
        f"Across all non-baseline evaluations, models beat random_walk "
        f"{beats}/{total} times by RMSE; random_walk itself is the "
        f"per-ticker winner {n_best_baseline} times.",
        f"Mean directional accuracy (non-baseline): {da_overall:.4f} "
        f"(0.5 = coin flip).",
        "Classical ML does NOT consistently beat the zero-return baseline; "
        "results are mixed across stocks, and where models trail the "
        "baseline that is reported as-is above.",
        "Daily stock-return prediction is difficult; these statistics must "
        "NOT be interpreted as proof of profitable trading performance.",
        "MAPE is reported for completeness but is unstable for near-zero "
        "returns; RMSE/MAE remain the primary error metrics. R2 here is a "
        "variance-explained statistic, not a baseline comparison.",
        "WARNING: best_model_by_ticker.csv selects winners using TEST-set "
        "metrics for reporting only and must not be used for live model "
        "selection; live choice requires a fresh holdout.",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def run_evaluation(config_path: str | Path = "config.yaml",
                   models_dir: str | Path = MODELS_DIR,
                   output_dir: str | Path = EVAL_DIR) -> dict:
    """Evaluate existing Task 2C outputs (offline; never trains)."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    tickers: list[str] = list(
        load_config(config_path)["universe"]["tickers"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stored, preds = load_model_outputs(models_dir)  # loud if missing
    metrics = validate_outputs(stored, preds, tickers)  # loud if bad
    comparison = baseline_comparison(metrics)
    best = best_model_by_ticker(metrics, comparison)
    aggregate = aggregate_summary(metrics, comparison)

    metrics.to_csv(out_dir / "evaluation_metrics.csv", index=False)
    comparison.to_csv(out_dir / "baseline_comparison.csv", index=False)
    best.to_csv(out_dir / "best_model_by_ticker.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_model_summary.csv", index=False)
    plots = make_plots(metrics, comparison, aggregate, out_dir / "plots")
    write_summary(metrics, comparison, best, aggregate,
                  out_dir / "evaluation_summary.txt")

    print("=" * 72)
    print("TASK 2D COMPLETE — evaluation only (no retraining)")
    print(f"Outputs -> {out_dir} "
          f"(metric_source={metrics['metric_source'].unique().tolist()})")
    print("=" * 72)
    return {"output_dir": str(out_dir), "plots": plots,
            "n_tickers": metrics["ticker"].nunique()}


def main(config_path: str | Path = "config.yaml") -> dict:
    """CLI entry point: ``python -m src.evaluate``."""
    return run_evaluation(config_path=config_path)


if __name__ == "__main__":
    main()
