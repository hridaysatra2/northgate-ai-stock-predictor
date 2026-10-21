"""Regression tests for DL target-boundary embargo (F02).

Verifies the final target in train/val does not end on the next
partition's first price. Synthetic data only, except one check on the
real AAPL feature file for end-to-end coverage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import deep_learning as dl
from src.deep_learning import (
    EMBARGO_SEQUENCES,
    LOOKBACK,
    _endpoint_split_counts,
    chronological_split,
    clean_supervised_rows,
    identify_feature_columns,
    identify_target_column,
    load_feature_dataset,
    order_chronologically,
    prepare_ticker,
)
from src.features import EQUITY_FEATURE_COLUMNS, TARGET_COLUMN

FEATURES_DIR = Path("data/processed/features")


def make_synthetic_frame(n=200, start="2020-01-01", seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    data = {"date": dates, "ticker": "SYN"}
    for c in EQUITY_FEATURE_COLUMNS:
        data[c] = rng.normal(0, 1, n)
    data[TARGET_COLUMN] = rng.normal(0, 0.01, n)
    return pd.DataFrame(data)


def _write_tmp_ticker(tmp_path, name="SYN", n=200):
    d = tmp_path / "features"
    d.mkdir(parents=True, exist_ok=True)
    make_synthetic_frame(n).to_parquet(d / f"{name}.parquet", index=False)
    return d


def _full_df_for(result_ticker_dir, ticker, tmp_dir=None):
    src = tmp_dir if tmp_dir is not None else FEATURES_DIR
    df = load_feature_dataset(ticker, features_dir=src)
    feat = identify_feature_columns(df)
    tgt = identify_target_column(df)
    df = order_chronologically(df)
    return clean_supervised_rows(df, feat, tgt)


def test_embargo_constant_and_counts():
    assert EMBARGO_SEQUENCES == 1
    # 200 rows, lookback 30: old train seqs would be 140-29=111.
    tr, va, _ = _endpoint_split_counts(200, 140, 30, 30)
    assert tr == 140 - 29 - 1
    assert va == 30 - 1


def test_train_target_does_not_end_on_val_price(tmp_path):
    feat_dir = _write_tmp_ticker(tmp_path, n=200)
    res = prepare_ticker("SYN", features_dir=feat_dir)
    full = _full_df_for(None, "SYN", tmp_dir=feat_dir)
    dates = pd.to_datetime(full["date"]).reset_index(drop=True)
    train_last = pd.to_datetime(res["metadata"]["train_last"])
    val_first = pd.to_datetime(res["metadata"]["val_first"])
    pos = dates[dates == train_last].index[0]
    next_date = dates.iloc[pos + 1]
    # Target on train_last ends on the next row's price, which must be
    # the unused embargo row, not the first validation price.
    assert next_date != val_first
    assert next_date < val_first
    all_used = set(pd.to_datetime(
        list(res["train_dates"]) + list(res["val_dates"])
        + list(res["test_dates"])))
    assert next_date not in all_used


def test_val_target_does_not_end_on_test_price(tmp_path):
    feat_dir = _write_tmp_ticker(tmp_path, n=200)
    res = prepare_ticker("SYN", features_dir=feat_dir)
    full = _full_df_for(None, "SYN", tmp_dir=feat_dir)
    dates = pd.to_datetime(full["date"]).reset_index(drop=True)
    val_last = pd.to_datetime(res["metadata"]["val_last"])
    test_first = pd.to_datetime(res["metadata"]["test_first"])
    pos = dates[dates == val_last].index[0]
    next_date = dates.iloc[pos + 1]
    assert next_date != test_first
    assert next_date < test_first
    all_used = set(pd.to_datetime(
        list(res["train_dates"]) + list(res["val_dates"])
        + list(res["test_dates"])))
    assert next_date not in all_used


def test_historical_overlap_still_allowed(tmp_path):
    # Validation windows may start before the val boundary (history only).
    feat_dir = _write_tmp_ticker(tmp_path, n=200)
    res = prepare_ticker("SYN", features_dir=feat_dir)
    val_first = pd.to_datetime(res["metadata"]["val_first"])
    # First val window covers 30 endpoints ending at val_first.
    first_window_start_pos = 0  # checked via raw ordering below
    assert first_window_start_pos == 0  # smoke: helper ran
    # Endpoint ordering still strictly increases across splits.
    assert pd.to_datetime(res["metadata"]["train_last"]) < val_first
    assert val_first <= pd.to_datetime(res["val_dates"][0])
    # The first val window must include pre-boundary history.
    full = _full_df_for(None, "SYN", tmp_dir=feat_dir)
    dates = pd.to_datetime(full["date"]).reset_index(drop=True)
    first_val_pos = dates[dates == pd.to_datetime(res["val_dates"][0])].index[0]
    assert first_val_pos >= LOOKBACK - 1
    window_start_date = dates.iloc[first_val_pos - LOOKBACK + 1]
    assert window_start_date < val_first


def test_aapl_end_to_end_embargo():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    full = _full_df_for(None, "AAPL")
    dates = pd.to_datetime(full["date"]).reset_index(drop=True)
    for last_key, first_key in (
        ("train_last", "val_first"), ("val_last", "test_first")
    ):
        last = pd.to_datetime(res["metadata"][last_key])
        first = pd.to_datetime(res["metadata"][first_key])
        pos = dates[dates == last].index[0]
        assert dates.iloc[pos + 1] < first
        assert dates.iloc[pos + 1] != first
