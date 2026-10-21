"""Tests for src.deep_learning (Task 3A). No training, no internet.

Every leakage constraint is demonstrated with real assertions on data
values/dates, not merely by calling a function named "leakage_audit".
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src import deep_learning as dl
from src.deep_learning import (
    LOOKBACK,
    TrainOnlyStandardScaler,
    chronological_split,
    clean_supervised_rows,
    create_sequences,
    fit_feature_scaler,
    identify_feature_columns,
    identify_target_column,
    load_feature_dataset,
    order_chronologically,
    prepare_all_tickers,
    prepare_ticker,
    transform_features,
)
from src.features import EQUITY_FEATURE_COLUMNS, TARGET_COLUMN

FEATURES_DIR = Path("data/processed/features")
RAW_DIR = Path("data/raw")


def make_synthetic_frame(n=200, start="2020-01-01", seed=11):
    """Synthetic supervised frame with the real schema (business days)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    data = {"date": dates, "ticker": "SYN"}
    for c in EQUITY_FEATURE_COLUMNS:
        data[c] = rng.normal(0, 1, n)
    data[TARGET_COLUMN] = rng.normal(0, 0.01, n)
    return pd.DataFrame(data)


def cleaned_aapl():
    df = load_feature_dataset("AAPL", features_dir=FEATURES_DIR)
    feat = identify_feature_columns(df)
    tgt = identify_target_column(df)
    df = order_chronologically(df)
    return clean_supervised_rows(df, feat, tgt), feat, tgt


# 1. Sequence length is exactly 30.
def test_1_sequence_length_is_30():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    assert res["X_train"].shape[1] == LOOKBACK == 30
    assert res["X_val"].shape[1] == 30
    assert res["X_test"].shape[1] == 30
    assert res["metadata"]["lookback"] == 30
    # Pure helper also respects lookback.
    df = make_synthetic_frame(100)
    feat = identify_feature_columns(df)
    X, y, d = create_sequences(
        df[feat].to_numpy(float), df[TARGET_COLUMN].to_numpy(float),
        df["date"], lookback=30,
    )
    assert X.shape[1] == 30 and X.shape[2] == len(feat)


# 2. Sequence timestamps are chronological.
def test_2_sequence_timestamps_chronological():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    for key in ("train_dates", "val_dates", "test_dates"):
        dates = pd.to_datetime(pd.Series(res[key]))
        assert dates.is_monotonic_increasing
    # Windows are consecutive ordered rows: spot-check first train window
    # against the full ordered history.
    df, feat, tgt = cleaned_aapl()
    scaler = res["scaler"]
    full_scaled = transform_features(df, scaler, feat)
    np.testing.assert_allclose(res["X_train"][0], full_scaled[0:30], rtol=1e-12)
    np.testing.assert_allclose(res["X_train"][1], full_scaled[1:31], rtol=1e-12)


# 3. Sequence ending at t maps to target_t.
def test_3_endpoint_maps_to_target():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    df, feat, tgt = cleaned_aapl()
    targets = df[tgt].to_numpy(float)
    dates = pd.to_datetime(df["date"])
    for X, y, d in (
        (res["X_train"], res["y_train"], res["train_dates"]),
        (res["X_val"], res["y_val"], res["val_dates"]),
        (res["X_test"], res["y_test"], res["test_dates"]),
    ):
        # Every endpoint date must exist in the source frame and its label
        # must equal that row's target exactly.
        for k in (0, len(y) // 2, len(y) - 1):
            pos = dates[dates == pd.to_datetime(d[k])].index
            assert len(pos) == 1
            assert y[k] == pytest.approx(targets[pos[0]])
        # Final observation of each window equals the endpoint row's
        # (scaled) features.
        full_scaled = transform_features(df, res["scaler"], feat)
        for k in (0, len(y) - 1):
            pos = dates[dates == pd.to_datetime(d[k])].index[0]
            np.testing.assert_allclose(X[k][-1], full_scaled[pos], rtol=1e-12)


# 4. No t+1 observation appears inside a sequence predicting t.
def test_4_no_tplus1_inside_sequence():
    df, feat, tgt = cleaned_aapl()
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    dates = pd.to_datetime(df["date"])
    raw_vals = df[feat].to_numpy(float)
    # The last window position of the first training sequence must equal
    # row j (endpoint), never row j+1.
    j = LOOKBACK - 1  # endpoint position of first sequence
    assert pd.to_datetime(res["train_dates"][0]) == dates.iloc[j]
    # Unscale the stored window's last row and compare to raw row j,
    # and assert it differs from row j+1 (shock is real data, not equal).
    mean = res["scaler"].mean_
    scale = res["scaler"].scale_
    last_unscaled = res["X_train"][0][-1] * scale + mean
    np.testing.assert_allclose(last_unscaled, raw_vals[j], rtol=1e-10)
    assert not np.allclose(last_unscaled, raw_vals[j + 1], rtol=1e-6)
    # Stronger: corrupting row j+1's FEATURES leaves the sequence ending
    # at j bit-identical (proved again end-to-end in test 11).


# 5. No future row appears in an earlier sequence.
def test_5_no_future_row_in_earlier_sequence():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    df, feat, _ = cleaned_aapl()
    dates = pd.to_datetime(df["date"])
    # Endpoint order across the whole dataset is strictly increasing.
    all_dates = pd.to_datetime(
        list(res["train_dates"]) + list(res["val_dates"]) + list(res["test_dates"])
    )
    assert all_dates.is_monotonic_increasing
    assert len(set(all_dates)) == len(all_dates)
    # Every window's observations are at or before its endpoint: the last
    # training window ends exactly at the last training endpoint, which is
    # strictly before any validation/test endpoint.
    assert pd.to_datetime(res["train_dates"].max()) < pd.to_datetime(
        res["val_dates"].min())
    # Window start of first sequence is the first available row (no future).
    assert pd.to_datetime(res["train_dates"][0]) == dates.iloc[LOOKBACK - 1]
    # Train windows never reach into val/test rows: the last train window
    # ends one row before the train-block end (1-sequence embargo, since
    # the target on the last train row ends on the first val price).
    df_len = len(df)
    n_train_rows = res["metadata"]["n_rows_train"]
    last_train_end = dates[dates == pd.to_datetime(res["train_dates"][-1])].index[0]
    assert last_train_end == n_train_rows - 2
    assert last_train_end < df_len - 1  # future rows exist but are excluded
    # The embargo endpoint (last train row) is in no split.
    embargo_date = dates.iloc[n_train_rows - 1]
    assert embargo_date not in set(pd.to_datetime(all_dates))


# 6. Train/validation/test prediction timestamps are chronological
#    and non-overlapping.
def test_6_splits_chronological_nonoverlapping():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    tr = set(pd.to_datetime(pd.Series(res["train_dates"])))
    va = set(pd.to_datetime(pd.Series(res["val_dates"])))
    te = set(pd.to_datetime(pd.Series(res["test_dates"])))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert max(tr) < min(va) and max(va) < min(te)
    # Split proportions on target rows are 70/15/15 (integer truncation).
    m = res["metadata"]
    n = m["n_rows_total"]
    assert m["n_rows_train"] == int(n * 0.70)
    assert m["n_rows_val"] == int(n * 0.15)
    assert m["n_rows_train"] + m["n_rows_val"] + m["n_rows_test"] == n


# 7. Training ends before validation.
def test_7_training_ends_before_validation():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    assert pd.to_datetime(res["metadata"]["train_last"]) < pd.to_datetime(
        res["metadata"]["val_first"])
    df = make_synthetic_frame(200)
    tr, va, _ = chronological_split(df)
    assert tr["date"].max() < va["date"].min()


# 8. Validation ends before test.
def test_8_validation_ends_before_test():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    assert pd.to_datetime(res["metadata"]["val_last"]) < pd.to_datetime(
        res["metadata"]["test_first"])
    df = make_synthetic_frame(200)
    _, va, te = chronological_split(df)
    assert va["date"].max() < te["date"].min()


# 9. Scaler is fitted only on training data.
def test_9_scaler_fitted_only_on_training():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    df, feat, _ = cleaned_aapl()
    train_df, _, _ = chronological_split(df)
    expected_mean = train_df[feat].to_numpy(float).mean(axis=0)
    expected_std = train_df[feat].to_numpy(float).std(axis=0, ddof=0)
    expected_std = np.where(expected_std == 0.0, 1.0, expected_std)
    np.testing.assert_allclose(res["scaler"].mean_, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(res["scaler"].scale_, expected_std, rtol=1e-12)
    # Full-sample statistics differ (otherwise the test would be vacuous).
    full_mean = df[feat].to_numpy(float).mean(axis=0)
    assert not np.allclose(full_mean, expected_mean)


# 10. Validation/test are transformed using the training scaler.
def test_10_val_test_use_training_scaler():
    res = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    df, feat, _ = cleaned_aapl()
    dates = pd.to_datetime(df["date"])
    raw = df[feat].to_numpy(float)
    mean, scale = res["scaler"].mean_, res["scaler"].scale_
    # Manual transform of the first validation endpoint row must equal the
    # last step of the first validation window.
    pos = dates[dates == pd.to_datetime(res["val_dates"][0])].index[0]
    manual = (raw[pos] - mean) / scale
    np.testing.assert_allclose(res["X_val"][0][-1], manual, rtol=1e-12)
    pos_t = dates[dates == pd.to_datetime(res["test_dates"][0])].index[0]
    manual_t = (raw[pos_t] - mean) / scale
    np.testing.assert_allclose(res["X_test"][0][-1], manual_t, rtol=1e-12)
    # And it must NOT equal a transform using val/test's own statistics.
    val_df = df.iloc[res["metadata"]["n_rows_train"]:
                     res["metadata"]["n_rows_train"] + res["metadata"]["n_rows_val"]]
    own_mean = val_df[feat].to_numpy(float).mean(axis=0)
    assert not np.allclose(mean, own_mean)


# 11. Changing a future test value does not change any training sequence.
def test_11_future_test_change_keeps_train_identical(tmp_path):
    df = make_synthetic_frame(200)
    feat = identify_feature_columns(df)
    tgt = identify_target_column(df)
    df = order_chronologically(df)
    df = clean_supervised_rows(df, feat, tgt)
    train_df, _, _ = chronological_split(df)
    scaler = fit_feature_scaler(train_df, feat)
    full = transform_features(df, scaler, feat)
    X_base, _, _ = create_sequences(
        full, df[tgt].to_numpy(float), df["date"], lookback=30)
    n_train_seq = len(train_df) - 30 + 1
    X_train_base = X_base[:n_train_seq]

    shocked = df.copy()
    # Corrupt the LAST test rows' features drastically.
    shocked.loc[shocked.index[-5:], feat] *= 999.0
    # Scaler is still fitted on the ORIGINAL training rows only.
    full_shocked = transform_features(shocked, scaler, feat)
    X_shocked, _, _ = create_sequences(
        full_shocked, shocked[tgt].to_numpy(float), shocked["date"], lookback=30)
    np.testing.assert_array_equal(X_shocked[:n_train_seq], X_train_base)

    # End-to-end through prepare_ticker with a copied feature file.
    src = FEATURES_DIR / "AAPL.parquet"
    tmp_feat = tmp_path / "features"
    tmp_feat.mkdir()
    base = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    raw = pd.read_parquet(src)
    raw = raw.sort_values("date").reset_index(drop=True)
    n = len(raw)
    n_train_rows = int(n * 0.70)
    raw.loc[raw.index[n_train_rows + 10:], feat[0]] *= -777.0
    raw.to_parquet(tmp_feat / "AAPL.parquet", index=False)
    shocked_res = prepare_ticker("AAPL", features_dir=tmp_feat)
    np.testing.assert_array_equal(shocked_res["X_train"], base["X_train"])
    np.testing.assert_array_equal(shocked_res["y_train"], base["y_train"])


# 12. Repeated execution is deterministic.
def test_12_deterministic():
    a = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    b = prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        np.testing.assert_array_equal(a[k], b[k])
    pdt.assert_series_equal(
        pd.to_datetime(pd.Series(a["train_dates"])),
        pd.to_datetime(pd.Series(b["train_dates"])), check_names=False)
    assert a["metadata"] == b["metadata"]


def _hash_files(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in sorted(paths)}


# 13. Raw market files remain unchanged.
def test_13_raw_untouched(tmp_path):
    before = _hash_files(RAW_DIR.rglob("*.parquet"))
    assert len(before) >= 12
    prepare_ticker("AAPL", features_dir=FEATURES_DIR)
    prepare_all_tickers(
        tickers=["AAPL", "MSFT"], features_dir=FEATURES_DIR,
        output_dir=tmp_path / "dl")
    after = _hash_files(RAW_DIR.rglob("*.parquet"))
    assert before == after


# 14. Existing feature files remain unchanged.
def test_14_features_untouched(tmp_path):
    before = _hash_files(FEATURES_DIR.glob("*.parquet"))
    assert len(before) == 12
    meta_before = _hash_files([FEATURES_DIR / "feature_metadata.csv"])
    prepare_all_tickers(
        tickers=["AAPL", "CAT"], features_dir=FEATURES_DIR,
        output_dir=tmp_path / "dl")
    after = _hash_files(FEATURES_DIR.glob("*.parquet"))
    meta_after = _hash_files([FEATURES_DIR / "feature_metadata.csv"])
    assert before == after
    assert meta_before == meta_after


# 15. Missing/invalid required data fails loudly.
def test_15_missing_ticker_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_ticker("NOPE", features_dir=FEATURES_DIR)
    with pytest.raises(FileNotFoundError):
        load_feature_dataset("NOPE", features_dir=tmp_path)


def test_15_no_target_loud():
    # VIX diagnostics file intentionally lacks the supervised target.
    with pytest.raises(ValueError, match="target"):
        prepare_ticker("VIX", features_dir=FEATURES_DIR)


def test_15_missing_columns_loud():
    df = make_synthetic_frame(100).drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="target"):
        identify_target_column(df)
    with pytest.raises(ValueError, match="target"):
        prepare_ticker("VIX", features_dir=FEATURES_DIR)
    bad = make_synthetic_frame(100).drop(columns=EQUITY_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="feature"):
        identify_feature_columns(bad)


def test_15_too_few_rows_loud():
    df = make_synthetic_frame(20)  # fewer than lookback=30
    feat = identify_feature_columns(df)
    with pytest.raises(ValueError, match="lookback|not enough"):
        create_sequences(
            df[feat].to_numpy(float), df[TARGET_COLUMN].to_numpy(float),
            df["date"], lookback=30)
    tiny = make_synthetic_frame(40)
    with pytest.raises(ValueError, match="not enough rows"):
        chronological_split(tiny.head(5))
    # Unfitted scaler must refuse to transform rather than guess.
    with pytest.raises(ValueError, match="not been fitted"):
        TrainOnlyStandardScaler().transform(np.zeros((3, 2)))


def test_15_nonfinite_features_loud():
    df = make_synthetic_frame(120)
    feat = identify_feature_columns(df)
    df.loc[df.index[50], feat[0]] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        clean_supervised_rows(df, feat, TARGET_COLUMN)


# Extra: output files are reproducible and complete.
def test_output_files_complete(tmp_path):
    out = prepare_all_tickers(
        tickers=["AAPL", "MSFT"], features_dir=FEATURES_DIR,
        output_dir=tmp_path / "dl")
    assert out["n_tickers"] == 2
    for t in ("AAPL", "MSFT"):
        npz = tmp_path / "dl" / f"{t}.npz"
        meta = tmp_path / "dl" / f"{t}_metadata.json"
        assert npz.exists() and meta.exists()
        z = np.load(npz)
        assert set(z.files) >= {"X_train", "y_train", "train_dates",
                                "X_val", "y_val", "val_dates",
                                "X_test", "y_test", "test_dates",
                                "feature_names"}
        assert z["X_train"].shape[1:] == (30, z["X_train"].shape[2])
        assert len(z["X_train"]) == len(z["y_train"]) == len(z["train_dates"])
        m = json.loads(meta.read_text())
        for k in ("ticker", "lookback", "n_features", "feature_names",
                  "n_train", "n_val", "n_test",
                  "train_first", "train_last", "val_first", "val_last",
                  "test_first", "test_last", "scaler"):
            assert k in m
        assert m["scaler"]["fitted_on"] == "training_only"
        assert m["n_features"] == len(m["feature_names"]) == 30
    summary = pd.read_csv(tmp_path / "dl" / "dataset_summary.csv")
    assert set(summary["ticker"]) == {"AAPL", "MSFT"}
