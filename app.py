"""Northgate AI Stock Predictor — Research Dashboard (Task 5A).

Research / decision-support interface ONLY. Read-only with respect to all
research artifacts: never retrains models, never optimizes, never writes
CSVs/Parquet/checkpoints/config. All figures are historical evaluations
from precomputed artifacts; nothing here is a live forecast or advice.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard_data import (
    ARTIFACT_PATHS,
    BACKTEST_ASSUMPTIONS,
    BENCHMARK_LABEL,
    BENCHMARK_TICKER,
    DISCLAIMER,
    HISTORICAL_BACKTEST_LABEL,
    LIVE_RECOMMENDATIONS_UNAVAILABLE,
    LIVE_UNAVAILABLE_MSG,
    NAV_PAGES,
    RESEARCH_CUTOFF,
    RESEARCH_START,
    RISK_REGIME_TICKER,
    SYNTHETIC_DEMO_LABEL,
    TICKERS,
    artifact_status,
    clip_to_cutoff,
    compute_drawdown,
    filter_real_recommendations,
    format_directional_accuracy,
    get_artifact_paths,
    get_live_recommendation_status,
    get_research_cutoff,
    is_synthetic_record,
    label_demo_frame,
    load_adf_results,
    load_csv_artifact,
    load_feature_frame,
    load_json_artifact,
    load_raw_prices,
    load_text_artifact,
    normalized_price,
    summarize_stock_history,
)

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows pure-helper imports in tests
    st = None  # type: ignore

__all__ = [
    "NAV_PAGES",
    "TICKERS",
    "RESEARCH_CUTOFF",
    "DISCLAIMER",
    "HISTORICAL_BACKTEST_LABEL",
    "SYNTHETIC_DEMO_LABEL",
    "LIVE_RECOMMENDATIONS_UNAVAILABLE",
    "get_artifact_paths",
    "main",
    "render_overview",
    "render_stock_analysis",
    "render_model_performance",
    "render_portfolio_optimization",
    "render_portfolio_backtest",
    "render_recommendations",
    "render_methodology",
]


# ---------------------------------------------------------------------------
# Cached read-only loaders (Streamlit cache; defensive copies on use)
# ---------------------------------------------------------------------------

def _cached_csv(key: str):
    if st is None:
        return load_csv_artifact(key)

    @st.cache_data(show_spinner=False)
    def _load(k: str):
        return load_csv_artifact(k)

    df = _load(key)
    return None if df is None else df.copy(deep=True)


def _cached_json(key: str):
    if st is None:
        return load_json_artifact(key)

    @st.cache_data(show_spinner=False)
    def _load(k: str):
        return load_json_artifact(k)

    return _load(key)


def _cached_text(key: str):
    if st is None:
        return load_text_artifact(key)

    @st.cache_data(show_spinner=False)
    def _load(k: str):
        return load_text_artifact(k)

    return _load(key)


def _cached_raw(ticker: str):
    if st is None:
        return load_raw_prices(ticker)

    @st.cache_data(show_spinner=False)
    def _load(t: str):
        return load_raw_prices(t)

    df = _load(ticker)
    return None if df is None else df.copy(deep=True)


def _cached_features(ticker: str):
    if st is None:
        return load_feature_frame(ticker)

    @st.cache_data(show_spinner=False)
    def _load(t: str):
        return load_feature_frame(t)

    df = _load(ticker)
    return None if df is None else df.copy(deep=True)


def _cached_adf():
    if st is None:
        return load_adf_results()

    @st.cache_data(show_spinner=False)
    def _load():
        return load_adf_results()

    df, reason = _load()
    return (None if df is None else df.copy(deep=True)), reason


def _missing(key: str) -> str:
    rel = ARTIFACT_PATHS.get(key, key)
    return f"Unavailable — expected artifact not found: `{rel}` (no data fabricated)."


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def render_overview():
    st.header("Overview — quantitative research terminal")
    st.info(DISCLAIMER)
    st.warning(LIVE_UNAVAILABLE_MSG)

    ds = _cached_csv("dataset_summary")
    n_obs = None
    if ds is not None and "n_rows_total" in ds.columns:
        n_obs = int(ds["n_rows_total"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Research cutoff", RESEARCH_CUTOFF)
    c2.metric("Equities", str(len(TICKERS)))
    c3.metric("Benchmark", BENCHMARK_LABEL)
    c4.metric("Risk regime", RISK_REGIME_TICKER)
    c1.metric("Data period (requested)", f"{RESEARCH_START} → {RESEARCH_CUTOFF}")
    c2.metric("Total feature rows (10 equities + benchmark)", f"{n_obs:,}" if n_obs else "Unavailable")

    st.subheader("Universe")
    st.write(", ".join(TICKERS) + f"  ·  benchmark {BENCHMARK_TICKER}  ·  VIX")

    st.subheader("High-level research findings (from frozen artifacts — historical only)")
    summary = _cached_text("evaluation_summary_txt")
    if summary:
        st.code(summary, language="text")
    else:
        st.warning(_missing("evaluation_summary_txt"))
    agg = _cached_csv("aggregate_summary")
    if agg is not None:
        st.dataframe(agg.copy(deep=True), use_container_width=True)
        st.caption("Historical test-set evaluation only. Not a live model ranking; no profitability claim.")
    else:
        st.warning(_missing("aggregate_summary"))

    st.subheader("Artifact availability")
    status = artifact_status()
    st.dataframe(
        pd.DataFrame([{"artifact": k, "status": v} for k, v in status.items()]),
        use_container_width=True,
    )


def render_stock_analysis():
    st.header("Stock Analysis — historical research statistics")
    st.caption("Historical adjusted-close research statistics. Historical patterns do not predict future prices.")
    ticker = st.selectbox("Ticker", TICKERS, index=0)
    raw = _cached_raw(ticker)
    feat = _cached_features(ticker)

    if raw is None:
        st.warning(_missing(f"data/raw/market/{ticker}.parquet"))
        return
    df = raw.copy(deep=True)

    stats = summarize_stock_history(df)
    if stats:
        cols = st.columns(4)
        cols[0].metric("Observations", f"{stats['n_observations']:,}")
        cols[1].metric("Last adj close", f"{stats['last_adj_close']:.2f}")
        cols[2].metric("Skewness (log ret)", f"{stats['skewness']:.3f}")
        cols[3].metric("Kurtosis (log ret)", f"{stats['kurtosis']:.3f}")
        st.caption(f"Period: {stats['first_date']} → {stats['last_date']} (historical).")
        st.json({k: v for k, v in stats.items()})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["adj_close"], mode="lines", name=f"{ticker} adj close"))
    if len(df) >= 20:
        fig.add_trace(go.Scatter(x=df["date"], y=df["adj_close"].rolling(20).mean(),
                                 mode="lines", name="SMA 20"))
    if len(df) >= 50:
        fig.add_trace(go.Scatter(x=df["date"], y=df["adj_close"].rolling(50).mean(),
                                 mode="lines", name="SMA 50"))
    fig.update_layout(title=f"{ticker} — adjusted close (historical)",
                      xaxis_title="Date", yaxis_title="Adj close", legend_title="Series",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Normalized comparison vs benchmark
    bench = _cached_raw(BENCHMARK_TICKER)
    if bench is not None:
        merged = pd.DataFrame({"date": pd.to_datetime(df["date"]), ticker: pd.to_numeric(df["adj_close"], errors="coerce")})
        b = pd.DataFrame({"date": pd.to_datetime(bench["date"]), BENCHMARK_TICKER: pd.to_numeric(bench["adj_close"], errors="coerce")})
        m = pd.merge(merged, b, on="date", how="inner")
        if len(m) > 0:
            m[ticker + " (rebased 100)"] = normalized_price(m[ticker])
            m[BENCHMARK_TICKER + " (rebased 100)"] = normalized_price(m[BENCHMARK_TICKER])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=m["date"], y=m[ticker + " (rebased 100)"], mode="lines", name=ticker))
            fig2.add_trace(go.Scatter(x=m["date"], y=m[BENCHMARK_TICKER + " (rebased 100)"], mode="lines", name=BENCHMARK_LABEL))
            fig2.update_layout(title=f"{ticker} vs {BENCHMARK_LABEL} — normalized price (historical, rebased 100)",
                               xaxis_title="Date", yaxis_title="Rebased price", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Benchmark series unavailable — normalized comparison skipped (no substitute used).")

    # Rolling volatility + return distribution + drawdown
    log_ret = pd.Series(dtype=float)
    if feat is not None and "log_return_1d" in feat.columns:
        r = pd.to_numeric(feat["log_return_1d"], errors="coerce").dropna()
        log_ret = r
        vol20 = r.rolling(20).std(ddof=1) * (252 ** 0.5)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=feat["date"].iloc[r.index] if "date" in feat.columns else r.index,
                                  y=vol20, mode="lines", name="Rolling vol 20d (ann.)"))
        fig3.update_layout(title=f"{ticker} — rolling volatility, 20d annualized (historical)",
                           xaxis_title="Date", yaxis_title="Annualized volatility")
        st.plotly_chart(fig3, use_container_width=True)
    if len(log_ret) > 0:
        fig4 = px.histogram(x=log_ret, nbins=60, labels={"x": "Daily log return"},
                            title=f"{ticker} — daily log-return distribution (historical)")
        st.plotly_chart(fig4, use_container_width=True)
        dd = compute_drawdown(df["adj_close"])
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df["date"], y=dd, mode="lines", name="Drawdown"))
        fig5.update_layout(title=f"{ticker} — drawdown of adj-close value curve (historical)",
                           xaxis_title="Date", yaxis_title="Drawdown (fraction)")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Return/volatility series unavailable for this ticker.")

    st.subheader("Stationarity (ADF) — Task 2A, historical only")
    adf_df, adf_reason = load_adf_results() if st is None else _cached_adf()
    if adf_df is not None:
        st.dataframe(adf_df.copy(deep=True), use_container_width=True)
        st.caption("ADF statistic and p-value from frozen Task 2A results. High p-value fails to "
                   "reject the unit-root null; low p-value is evidence against the unit-root null. "
                   "No probability-of-non-stationarity claim is made.")
    else:
        st.info(adf_reason)


def render_model_performance():
    st.header("Model Performance — historical test evaluation only")
    st.caption("Classical (Task 2D) and deep-learning (Task 3B) test-set metrics. "
               "Do NOT interpret as live-trading rankings; no profitability claim.")
    eval_df = _cached_csv("evaluation_metrics")
    agg = _cached_csv("aggregate_summary")
    best = _cached_csv("best_by_ticker")
    base = _cached_csv("baseline_comparison")
    dl = _cached_csv("dl_metrics")

    if agg is not None:
        a = agg.copy(deep=True)
        fig = px.bar(a, x="model", y="mean_rmse", color="model",
                     title="Mean test RMSE by model — historical evaluation only (log-return units)",
                     labels={"mean_rmse": "Mean RMSE (log-return units)", "model": "Model"})
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(a, x="model", y="mean_directional_accuracy", color="model",
                      title="Mean directional accuracy by model — historical only (proportion; 0.50 is coin flip)",
                      labels={"mean_directional_accuracy": "Mean directional accuracy (proportion)"})
        fig2.update_yaxes(range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(a, use_container_width=True)
        st.caption("RMSE/MAE are in log-return units; R² is a variance-explained statistic, not a baseline "
                   "comparison; MAPE is unstable near zero returns. Directional accuracy is a proportion "
                   f"(e.g. 0.509 = {format_directional_accuracy(0.509)}); values near 0.50 are coin-flip "
                   "performance, not predictive superiority. No model is 'best for trading'.")
    else:
        st.warning(_missing("aggregate_summary"))

    if eval_df is not None:
        e = eval_df.copy(deep=True)
        model = st.selectbox("Metric view — model", sorted(e["model"].unique().tolist()))
        sub = e[e["model"] == model].copy(deep=True)
        fig = px.bar(sub, x="ticker", y="rmse", color="ticker",
                     title=f"Per-ticker test RMSE — {model} (historical, log-return units)",
                     labels={"rmse": "RMSE (log-return units)"})
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(e, use_container_width=True)
        st.caption("Random Walk baseline (zero-return forecast) is included in every comparison. "
                   "Classical ML = linear_regression, ridge, random_forest, xgboost, svr; "
                   "Deep Learning = lstm, gru, bilstm, transformer (separate table below).")
    else:
        st.warning(_missing("evaluation_metrics"))

    if base is not None:
        st.subheader("Baseline comparison (historical)")
        st.dataframe(base.copy(deep=True), use_container_width=True)
    if best is not None:
        st.subheader("Best test-set RMSE — historical evaluation only")
        st.warning("Test-selected results must not be used as a live model-selection mechanism (Task 4F).")
        st.dataframe(best.copy(deep=True), use_container_width=True)
    if dl is not None:
        st.subheader("Deep learning vs classical (historical)")
        d = dl.copy(deep=True)
        fig = px.bar(d, x="model", y="rmse", color="ticker", barmode="group",
                     title="DL test RMSE by architecture and ticker — historical only (log-return units)",
                     labels={"rmse": "RMSE (log-return units)"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(d, use_container_width=True)
        st.caption("DL did not consistently improve out-of-sample RMSE over the Random Walk baseline (see methodology).")
    else:
        st.warning(_missing("dl_metrics"))


def render_portfolio_optimization():
    st.header("Portfolio Optimization — in-sample illustration (Task 4C)")
    st.caption("Optimization inputs are historical/in-sample assumptions (sample mean/covariance). "
               "The efficient frontier is not future guaranteed performance.")
    key = _cached_csv("key_portfolios")
    frontier = _cached_csv("efficient_frontier")
    mc = _cached_csv("monte_carlo")
    meta = _cached_json("frontier_metadata")

    if meta:
        st.json({k: meta[k] for k in ("n_obs", "date_min", "date_max", "disclaimer",
                                      "estimator", "risk_free_rate") if k in meta})

    if mc is not None and frontier is not None:
        m = mc.copy(deep=True)
        f = frontier.copy(deep=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["volatility"], y=m["expected_return"], mode="markers",
                                 marker=dict(size=3, opacity=0.35), name="Monte Carlo portfolios",
                                 hovertemplate="vol=%{x:.4f}<br>ret=%{y:.5f}"))
        fig.add_trace(go.Scatter(x=f["volatility"], y=f["expected_return"], mode="lines+markers",
                                 name="Efficient frontier", line=dict(width=3)))
        if key is not None:
            k = key.copy(deep=True)
            fig.add_trace(go.Scatter(x=k["volatility"], y=k["expected_return"], mode="markers+text",
                                     marker=dict(size=12), text=k["portfolio"], textposition="top center",
                                     name="Key portfolios"))
        fig.update_layout(title="Monte Carlo portfolio cloud + efficient frontier (historical, in-sample)",
                          xaxis_title="Volatility (daily)", yaxis_title="Expected return (daily)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(_missing("monte_carlo_portfolios.csv / efficient_frontier.csv"))

    if key is not None:
        k = key.copy(deep=True)
        st.subheader("Key portfolios (historical assumptions)")
        st.caption("Equal Weight vs Minimum Variance vs Maximum Sharpe under in-sample/sample-mean-covariance "
                   "assumptions. The S&P 500 (GSPC) benchmark is evaluation-only and is never a portfolio "
                   "constituent; weight columns cover the 10 equities only.")
        st.dataframe(k, use_container_width=True)
        weight_cols = [c for c in k.columns if c in TICKERS]
        if weight_cols:
            sel = st.selectbox("Inspect weights — portfolio", k["portfolio"].tolist())
            row = k[k["portfolio"] == sel].iloc[0]
            wdf = pd.DataFrame({"asset": weight_cols, "weight": [float(row[c]) for c in weight_cols]})
            fig = px.bar(wdf, x="asset", y="weight", title=f"Weights — {sel} (historical assumptions)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(wdf, use_container_width=True)
    else:
        st.warning(_missing("key_portfolios"))


def render_portfolio_backtest():
    st.header(HISTORICAL_BACKTEST_LABEL + " — Task 4D (historical simulation)")
    st.caption(BACKTEST_ASSUMPTIONS + " Do not interpret as a forecast; past returns do not imply future returns.")
    vals = _cached_csv("daily_values")
    metrics = _cached_csv("strategy_metrics")
    summary = _cached_json("backtest_summary")

    if summary:
        st.json({k: summary[k] for k in ("actual_first_date", "actual_last_date", "n_rebalance_dates",
                                         "n_trading_observations", "costs", "disclaimer") if k in summary})

    if vals is not None:
        v = clip_to_cutoff(vals.copy(deep=True))
        v["date"] = pd.to_datetime(v["date"])
        long_df = v.melt(id_vars="date", var_name="strategy", value_name="value")
        fig = px.line(long_df, x="date", y="value", color="strategy",
                      title="Cumulative portfolio value — historical rolling backtest (gross, no costs)",
                      labels={"value": "Portfolio value ($)", "date": "Date"})
        st.plotly_chart(fig, use_container_width=True)
        # Drawdown of equal-weight curve as illustration
        dd = compute_drawdown(v["equal_weight"] if "equal_weight" in v.columns else v.iloc[:, 1])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=v["date"], y=dd, mode="lines", name="Equal-weight drawdown"))
        fig2.update_layout(title="Drawdown — equal-weight backtest curve (historical)",
                           xaxis_title="Date", yaxis_title="Drawdown (fraction)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(_missing("daily_portfolio_values"))

    if metrics is not None:
        st.subheader("Strategy comparison (historical)")
        st.dataframe(metrics.copy(deep=True), use_container_width=True)
        st.caption("Units — cumulative_return: multiple of initial capital (e.g. 4.25 ≈ +425% growth of the "
                   "historical value curve, not an annualized rate); annualized_return_geo: geometric "
                   "annual rate; annualized_volatility: daily vol × √252; Sharpe/Sortino: ratios, not "
                   "percentages; maximum_drawdown: negative fraction; historical_var_daily: positive daily "
                   "loss at 95% confidence; total_turnover: sum of |weight changes|; "
                   "historical simulation only, gross of transaction costs.")
    else:
        st.warning(_missing("strategy_metrics"))


def render_recommendations():
    st.header("Recommendations — live state and synthetic demo")
    status = get_live_recommendation_status()
    st.error(status["status"])
    st.write(status["explanation"])
    st.warning(LIVE_UNAVAILABLE_MSG)
    st.caption("The Task 4E recommendation engine is implemented and tested, but production "
               "recommendations are not currently generated: no live forecast provider, "
               "no live price feed, no live risk feed, no validated live model-selection rule (Task 4F).")

    # Prove no real tickers leak through: the demo file contains only DEMO_* rows.
    demo = _cached_csv("demo_recommendations")
    st.subheader("Recommendation Engine Demo (optional, clearly separated)")
    st.warning(SYNTHETIC_DEMO_LABEL)
    if demo is None:
        st.info("Synthetic demo artifact not present — nothing to show. No recommendations fabricated.")
        return
    d = demo.copy(deep=True)
    real = filter_real_recommendations(d)
    st.write(f"Real-ticker rows in production view: **{len(real)}** (expected 0 — DEMO_* rows are never real).")
    labeled = label_demo_frame(d)
    st.dataframe(labeled, use_container_width=True)
    st.caption("Return-convention note (synthetic demo): `predicted_log_return` is the log-return forecast "
               "over (t, t+1]; `predicted_simple_return = exp(predicted_log_return) − 1` is the equivalent "
               "simple return; `predicted_price = current_price × exp(predicted_log_return)`. "
               "Neither is a realized return.")
    for _, row in labeled.iterrows():
        rec = row.to_dict()
        badge = "synthetic" if is_synthetic_record(rec) else "REAL — INVESTIGATE"
        st.write(f"`{row.get('ticker')}` → **{row.get('recommendation')}** · {SYNTHETIC_DEMO_LABEL} · ({badge})")
    st.caption("DEMO_* tickers are synthetic software-exercise rows and must never appear alongside real stocks "
               "without this unmistakable synthetic label.")


def render_methodology():
    st.header("Methodology & Limitations")
    st.subheader("Data")
    st.write("- yfinance; requested research range 2016-01-01 through 2026-08-31 (research cutoff 2026-08-31); "
             "actual available dates as documented per artifact; adjusted close usage.")
    st.subheader("Features")
    st.write("- Leakage-safe trailing features; 1-day-ahead log-return target "
             "`target_t = log(AdjClose_{t+1} / AdjClose_t)`.")
    st.subheader("Validation")
    st.write("- Chronological splits; walk-forward / TimeSeriesSplit; gap/embargo; "
             "DL chronological split; train-only scaling.")
    st.subheader("Models")
    st.write("- Linear Regression, Ridge, Random Forest, XGBoost, SVR, LSTM, GRU, BiLSTM, "
             "Transformer, Random Walk baseline.")
    st.subheader("Portfolio")
    st.write("- MPT; long-only; rolling optimization; 252-day lookback; monthly rebalancing.")
    st.subheader("Sentiment")
    st.error("Historical sentiment source unavailable — no substitute used (Task 3C).")
    news = _cached_text("news_availability")
    if news:
        st.code(news[:4000], language="text")
    st.subheader("Recommendations")
    st.write("- Rule-based; configurable thresholds; risk-aware; no live forecast provider.")
    reco = _cached_text("recommendation_methodology")
    if reco:
        st.code(reco[:4000], language="text")
    st.subheader("Limitations")
    st.write("- Model performance mixed; Random Walk baseline often competitive; "
             "DL did not consistently improve OOS RMSE; no profitability claims; "
             "no transaction costs in backtest; no live trading integration; "
             "no validated sentiment pipeline.")
    audit = _cached_text("integration_audit")
    if audit:
        with st.expander("Task 4F integration audit (excerpt)"):
            st.code(audit[:5000], language="text")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------

def main():
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is not installed. Install with: pip install streamlit plotly")
    st.set_page_config(page_title="Northgate AI — Research Dashboard", layout="wide")
    st.sidebar.title("Northgate AI")
    st.sidebar.caption("Research Dashboard")
    page = st.sidebar.radio("Navigation", NAV_PAGES, index=0)
    st.sidebar.caption(f"Research cutoff: {get_research_cutoff()}")
    st.sidebar.warning(LIVE_UNAVAILABLE_MSG)
    st.title("Northgate AI Stock Predictor")
    st.caption(DISCLAIMER)
    if page == "Overview":
        render_overview()
    elif page == "Stock Analysis":
        render_stock_analysis()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Portfolio Optimization":
        render_portfolio_optimization()
    elif page == "Portfolio Backtest":
        render_portfolio_backtest()
    elif page == "Recommendations":
        render_recommendations()
    elif page == "Methodology & Limitations":
        render_methodology()


if __name__ == "__main__":
    main()
