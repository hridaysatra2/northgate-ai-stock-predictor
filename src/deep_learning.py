"""Leakage-free sequence preparation for deep learning (Task 3A).

Preparation ONLY: builds 30-trading-day input sequences from the existing
Task 2B feature files under ``data/processed/features/`` for later
LSTM/GRU/BiLSTM/Transformer training. No neural networks are trained,
tuned, or evaluated here.

Target:
    target_t = log(AdjClose_{t+1} / AdjClose_t)

Prediction timestamp is t, so every input feature used to predict
target_t must be available no later than t:
    X[i] = observations from t-29 through t
    y[i] = target_t
The final observation in X[i] corresponds to t.

Leakage rules enforced:
  - Rows are chronologically ordered; split is 70/15/15 on prediction
    timestamps (target rows), never shuffled.
  - Sequences are sliding windows over the full ordered history, assigned
    to a split by their ENDPOINT t. A validation/test sequence may therefore
    contain historical observations from before its split boundary, but no
    validation/test TARGET ever enters training and no observation later
    than t ever enters the sequence predicting t.
  - One boundary sequence at the end of the train block and one at the
    end of the validation block are left unused (1-sequence embargo).
    The target on row t needs the price at t+1, so the sequence ending
    on the last train row (whose target ends on the first validation
    price) and the sequence ending on the last validation row (whose
    target ends on the first test price) are excluded from training
    and validation respectively. Historical window overlap is still
    allowed: validation/test windows may start before the boundary.
  - The feature scaler is fitted on TRAINING rows only, then applied to
    validation/test. Never refit on validation/test. Target is never scaled.
  - No forward-fill, no interpolation, no synthesis. Rows that cannot form
    a complete 30-day window or have an unavailable/invalid target are
    removed. Missing files/columns/rows raise loudly.

Outputs (``data/processed/deep_learning/``):
  - ``<TICKER>.npz``: X_train/y_train/train_dates, X_val/y_val/val_dates,
    X_test/y_test/test_dates, feature_names.
  - ``<TICKER>_metadata.json``: ticker, lookback, features, row/sequence
    counts, first/last prediction timestamp per split, scaler statistics.
  - ``dataset_summary.csv``: one row per ticker.
Raw market data, existing feature files, and Task 2C/2D outputs are only
ever read, never written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import EQUITY_FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

LOOKBACK = 30
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# Boundary sequences left unused so a target never ends on the next
# partition's first price (last train target -> first val price, last
# val target -> first test price). Windows may still start earlier.
EMBARGO_SEQUENCES = 1

FEATURES_DIR = Path("data/processed/features")
OUTPUT_DIR = Path("data/processed/deep_learning")

SUPERVISED_TICKERS = [
    "AAPL", "MSFT", "JPM", "JNJ", "PG",
    "XOM", "HD", "V", "CAT", "KO",
    "GSPC",
]


# ---------------------------------------------------------------------------
# Loading + schema
# ---------------------------------------------------------------------------

def sanitize_filename(ticker: str) -> str:
    """Strip '^' for safe filenames (``^GSPC`` -> ``GSPC``)."""
    return ticker.replace("^", "")


def load_feature_dataset(
    ticker: str,
    features_dir: str | Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Read-only load of one existing feature Parquet file.

    Raises FileNotFoundError when the file does not exist.
    """
    path = Path(features_dir) / (sanitize_filename(ticker) + ".parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"feature dataset missing for ticker {ticker!r}: {path}. "
            "Run the feature pipeline first; this module never fabricates data."
        )
    df = pd.read_parquet(path)
    return df.copy()


def identify_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns present in ``df`` per the existing feature schema.

    Returns the subset of ``EQUITY_FEATURE_COLUMNS`` found in ``df``
    (preserving canonical order). Raises ValueError when none are found.
    """
    cols = [c for c in EQUITY_FEATURE_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(
            "no known feature columns found; expected a subset of "
            f"EQUITY_FEATURE_COLUMNS, got columns={list(df.columns)}"
        )
    return cols


def identify_target_column(df: pd.DataFrame) -> str:
    """Return the target column name, raising loudly when absent.

    The VIX diagnostics file intentionally has no supervised target, so
    requesting sequences for it raises ValueError instead of silently
    producing bad labels.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"required target column {TARGET_COLUMN!r} missing "
            f"(columns={list(df.columns)}). Tickers without a supervised "
            "target (e.g. VIX diagnostics) cannot form prediction sequences."
        )
    return TARGET_COLUMN


def order_chronologically(
    df: pd.DataFrame, date_col: str = "date"
) -> pd.DataFrame:
    """Sort observations ascending by timestamp (deterministic)."""
    if date_col not in df.columns:
        raise ValueError(f"required date column {date_col!r} missing")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(date_col).reset_index(drop=True)


def clean_supervised_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Drop rows with unavailable/invalid target; fail loudly on bad input.

    - Raises ValueError when required columns are absent or the frame is
      empty after cleaning.
    - Raises ValueError when any feature value is non-finite (never filled
      or interpolated; such data must be fixed upstream, not hidden here).
    - Rows whose target is NaN/inf are removed (e.g. a trailing row whose
      t+1 price does not exist) and the removal count is logged.
    """
    missing = [c for c in feature_cols + [target_col, "date"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("feature dataset is empty; cannot build sequences")
    out = df.copy()
    y = pd.to_numeric(out[target_col], errors="coerce").to_numpy(dtype=float)
    bad_target = ~np.isfinite(y)
    if bool(np.all(bad_target)):
        raise ValueError("no valid (finite) target values; cannot build sequences")
    if bool(np.any(bad_target)):
        logger.info("dropping %d rows with invalid target", int(bad_target.sum()))
        out = out.loc[~bad_target].reset_index(drop=True)
    for c in feature_cols:
        v = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
        if not bool(np.all(np.isfinite(v))):
            raise ValueError(
                f"non-finite values in feature column {c!r}; refusing to "
                "fill/interpolate. Fix upstream feature data first."
            )
    if len(out) == 0:
        raise ValueError("no rows remain after removing invalid targets")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chronological split (on prediction timestamps / target rows)
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 chronological split on prediction-timestamp rows.

    The input must already be chronologically ordered (it is re-sorted
    here defensively). Never shuffles. Test rows are strictly later than
    validation rows, which are strictly later than training rows.
    """
    if not np.isclose(train_fraction + val_fraction + TEST_FRACTION, 1.0):
        raise ValueError("split fractions must sum to 1.0")
    df = order_chronologically(df, date_col=date_col)
    n = len(df)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"not enough rows for 70/15/15 split (n={n}, "
            f"train={n_train}, val={n_val}, test={n_test})"
        )
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    assert train[date_col].max() <= val[date_col].min()
    assert val[date_col].max() <= test[date_col].min()
    if not (train[date_col].max() < test[date_col].min()):
        raise ValueError("split is not strictly chronological")
    return train, val, test


# ---------------------------------------------------------------------------
# Train-only scaling (target never scaled)
# ---------------------------------------------------------------------------

class TrainOnlyStandardScaler:
    """Mean/std scaler fitted on training rows only.

    ``mean_``/``scale_`` come exclusively from the training portion.
    Validation/test rows are transformed with those frozen statistics via
    :meth:`transform` (never refit). Zero-variance features get scale 1.0
    to avoid division by zero while remaining exactly reproducible.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self.n_train_rows_: int | None = None

    def fit(self, train_values: np.ndarray, feature_names: list[str]):
        train_values = np.asarray(train_values, dtype=float)
        if train_values.ndim != 2 or train_values.shape[0] == 0:
            raise ValueError("need a non-empty 2D training matrix to fit scaler")
        if not np.all(np.isfinite(train_values)):
            raise ValueError("non-finite training values; refusing to fit scaler")
        self.mean_ = train_values.mean(axis=0)
        std = train_values.std(axis=0, ddof=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        self.feature_names_ = list(feature_names)
        self.n_train_rows_ = int(train_values.shape[0])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler has not been fitted yet")
        values = np.asarray(values, dtype=float)
        return (values - self.mean_) / self.scale_

    def fit_transform(
        self, train_values: np.ndarray, feature_names: list[str]
    ) -> np.ndarray:
        return self.fit(train_values, feature_names).transform(train_values)

    def to_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("scaler has not been fitted yet")
        return {
            "scaler": "standard",
            "fitted_on": "training_only",
            "n_train_rows": self.n_train_rows_,
            "feature_names": list(self.feature_names_ or []),
            "mean": [float(v) for v in np.asarray(self.mean_)],
            "scale": [float(v) for v in np.asarray(self.scale_)],
        }


def fit_feature_scaler(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> TrainOnlyStandardScaler:
    """Fit a feature scaler on TRAINING rows only (target never scaled)."""
    if TARGET_COLUMN in feature_cols:
        raise ValueError("target column must never be part of feature scaling")
    scaler = TrainOnlyStandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=float), feature_cols)
    return scaler


def transform_features(
    df: pd.DataFrame, scaler: TrainOnlyStandardScaler, feature_cols: list[str]
) -> np.ndarray:
    """Transform any split with the training-fitted scaler."""
    return scaler.transform(df[feature_cols].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def create_sequences(
    values: np.ndarray,
    targets: np.ndarray,
    dates: pd.Series | np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build contiguous ``lookback``-day sequences from one ordered block.

    For each valid endpoint position j >= lookback - 1:
        X[k] = values[j - lookback + 1 .. j]
        y[k] = targets[j]
        seq_dates[k] = dates[j]
    Raises ValueError when the block is too short or inputs are invalid.
    """
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    dates = np.asarray(pd.to_datetime(pd.Series(np.asarray(dates)).values))
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    n = values.shape[0]
    if not (len(targets) == n and len(dates) == n):
        raise ValueError("values/targets/dates length mismatch")
    if n < lookback:
        raise ValueError(
            f"not enough rows for lookback={lookback} (n={n})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite feature values; refusing to build sequences")
    if not np.all(np.isfinite(targets)):
        raise ValueError("non-finite targets; refusing to build sequences")
    n_seq = n - lookback + 1
    n_feat = values.shape[1]
    X = np.empty((n_seq, lookback, n_feat), dtype=float)
    for k in range(n_seq):
        X[k] = values[k:k + lookback]
    y = targets[lookback - 1:]
    seq_dates = dates[lookback - 1:]
    return X, y.astype(float), seq_dates


def _endpoint_split_counts(
    n_rows: int, n_train_rows: int, n_val_rows: int, lookback: int
) -> tuple[int, int, int]:
    """Sequence counts per split when endpoints decide membership.

    One boundary sequence at the end of the train block and one at the
    end of the validation block are left unused (1-sequence embargo):
    the target on the last train row ends on the first validation price,
    and the target on the last validation row ends on the first test
    price. Historical window overlap is still allowed.
    """
    first = lookback - 1
    n_seq_train = max(0, n_train_rows - first - EMBARGO_SEQUENCES)
    n_seq_val = max(0, n_val_rows - EMBARGO_SEQUENCES)
    n_seq_test = n_rows - n_train_rows - n_val_rows
    return n_seq_train, n_seq_val, n_seq_test


# ---------------------------------------------------------------------------
# Per-ticker + all-ticker preparation
# ---------------------------------------------------------------------------

def prepare_ticker(
    ticker: str,
    features_dir: str | Path = FEATURES_DIR,
    lookback: int = LOOKBACK,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> dict:
    """Prepare leakage-free scaled sequences for one ticker.

    Steps: load -> identify schema -> order -> drop invalid targets ->
    chronological 70/15/15 split on target rows -> fit scaler on training
    rows only -> transform the full ordered frame with that scaler ->
    sliding 30-day windows assigned to splits by endpoint t.

    Returns arrays, prediction timestamps, metadata, and the fitted scaler.
    Never writes to raw/feature/model/evaluation locations.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    df = load_feature_dataset(ticker, features_dir=features_dir)
    feature_cols = identify_feature_columns(df)
    target_col = identify_target_column(df)
    if target_col in feature_cols:
        raise ValueError("target column must not appear among feature columns")
    df = order_chronologically(df)
    df = clean_supervised_rows(df, feature_cols, target_col)

    n = len(df)
    if n < lookback:
        raise ValueError(
            f"not enough valid rows for lookback={lookback} (n={n})"
        )
    train_df, val_df, test_df = chronological_split(
        df, train_fraction=train_fraction, val_fraction=val_fraction
    )
    n_train_rows, n_val_rows = len(train_df), len(val_df)
    n_seq_train, n_seq_val, n_seq_test = _endpoint_split_counts(
        n, n_train_rows, n_val_rows, lookback
    )
    if min(n_seq_train, n_seq_val, n_seq_test) <= 0:
        raise ValueError(
            f"not enough rows for lookback={lookback} sequences "
            f"(rows={n}, train={n_train_rows}, val={n_val_rows}, "
            f"test={n - n_train_rows - n_val_rows})"
        )

    scaler = fit_feature_scaler(train_df, feature_cols)
    full_scaled = transform_features(df, scaler, feature_cols)
    full_targets = df[target_col].to_numpy(dtype=float)
    full_dates = pd.to_datetime(df["date"]).to_numpy()

    X_all, y_all, d_all = create_sequences(
        full_scaled, full_targets, full_dates, lookback=lookback
    )
    # Endpoint row position in the full frame: lookback-1 .. n-1.
    # Membership follows the endpoint's split (train endpoints first).
    # One boundary sequence after the train block and one after the
    # validation block are skipped (embargo): their targets end on the
    # next partition's first price.
    train_end = n_seq_train
    val_start = train_end + EMBARGO_SEQUENCES
    val_end = val_start + n_seq_val
    test_start = val_end + EMBARGO_SEQUENCES
    X_train, y_train, d_train = (
        X_all[:train_end], y_all[:train_end], d_all[:train_end])
    X_val, y_val, d_val = (
        X_all[val_start:val_end],
        y_all[val_start:val_end],
        d_all[val_start:val_end],
    )
    X_test, y_test, d_test = (
        X_all[test_start:],
        y_all[test_start:],
        d_all[test_start:],
    )
    assert len(X_train) == n_seq_train and len(X_val) == n_seq_val
    assert len(X_test) == n_seq_test
    assert d_train.max() < d_val.min() and d_val.max() < d_test.min()

    metadata = {
        "ticker": sanitize_filename(ticker),
        "lookback": int(lookback),
        "n_features": int(len(feature_cols)),
        "feature_names": list(feature_cols),
        "target_column": target_col,
        "target_definition": "log(AdjClose_{t+1} / AdjClose_t)",
        "split": ("chronological 70/15/15 on prediction timestamps "
                  "with 1-sequence embargo at train/val and val/test"),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": float(TEST_FRACTION),
        "n_rows_total": int(n),
        "n_rows_train": int(n_train_rows),
        "n_rows_val": int(n_val_rows),
        "n_rows_test": int(n - n_train_rows - n_val_rows),
        "n_train": int(n_seq_train),
        "n_val": int(n_seq_val),
        "n_test": int(n_seq_test),
        "train_first": str(pd.to_datetime(d_train.min()).date()),
        "train_last": str(pd.to_datetime(d_train.max()).date()),
        "val_first": str(pd.to_datetime(d_val.min()).date()),
        "val_last": str(pd.to_datetime(d_val.max()).date()),
        "test_first": str(pd.to_datetime(d_test.min()).date()),
        "test_last": str(pd.to_datetime(d_test.max()).date()),
        "sequence_shape": [int(n_seq_train), int(lookback), int(len(feature_cols))],
        "scaler": scaler.to_dict(),
    }
    return {
        "ticker": sanitize_filename(ticker),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "X_train": X_train, "y_train": y_train, "train_dates": d_train,
        "X_val": X_val, "y_val": y_val, "val_dates": d_val,
        "X_test": X_test, "y_test": y_test, "test_dates": d_test,
        "scaler": scaler,
        "metadata": metadata,
    }


def save_ticker_output(result: dict, output_dir: str | Path) -> dict[str, str]:
    """Save one ticker's sequences + metadata (no duplicate raw data)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker = result["ticker"]
    npz_path = out_dir / f"{ticker}.npz"
    meta_path = out_dir / f"{ticker}_metadata.json"
    np.savez_compressed(
        npz_path,
        X_train=result["X_train"], y_train=result["y_train"],
        train_dates=np.asarray(result["train_dates"], dtype="datetime64[ns]"),
        X_val=result["X_val"], y_val=result["y_val"],
        val_dates=np.asarray(result["val_dates"], dtype="datetime64[ns]"),
        X_test=result["X_test"], y_test=result["y_test"],
        test_dates=np.asarray(result["test_dates"], dtype="datetime64[ns]"),
        feature_names=np.asarray(result["feature_cols"], dtype=str),
    )
    meta_path.write_text(json.dumps(result["metadata"], indent=2), encoding="utf-8")
    return {"npz": str(npz_path), "metadata": str(meta_path)}


def load_config_tickers(config_path: str | Path = "config.yaml") -> list[str]:
    """Supervised ticker list from config (equities + benchmark, no VIX)."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    universe = cfg["universe"]
    tickers: list[str] = list(universe["tickers"])
    benchmark = universe.get("benchmark")
    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)
    return [sanitize_filename(t) for t in tickers]


def prepare_all_tickers(
    tickers: list[str] | None = None,
    config_path: str | Path = "config.yaml",
    features_dir: str | Path = FEATURES_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    lookback: int = LOOKBACK,
) -> dict:
    """Prepare and save leakage-free sequences for every supervised ticker.

    Deterministic and offline. Reads feature files only; writes one ``.npz``
    plus one metadata JSON per ticker and a ``dataset_summary.csv``.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if tickers is None:
        tickers = load_config_tickers(config_path)
    tickers = [sanitize_filename(t) for t in tickers]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for ticker in tickers:
        result = prepare_ticker(
            ticker, features_dir=features_dir, lookback=lookback
        )
        save_ticker_output(result, out_dir)
        m = result["metadata"]
        summary_rows.append({
            "ticker": m["ticker"],
            "lookback": m["lookback"],
            "n_features": m["n_features"],
            "n_rows_total": m["n_rows_total"],
            "n_train": m["n_train"], "n_val": m["n_val"], "n_test": m["n_test"],
            "train_first": m["train_first"], "train_last": m["train_last"],
            "val_first": m["val_first"], "val_last": m["val_last"],
            "test_first": m["test_first"], "test_last": m["test_last"],
        })
        logger.info(
            "%s: train=%d val=%d test=%d feats=%d (%s..%s | %s..%s | %s..%s)",
            ticker, m["n_train"], m["n_val"], m["n_test"], m["n_features"],
            m["train_first"], m["train_last"], m["val_first"], m["val_last"],
            m["test_first"], m["test_last"],
        )
    summary = pd.DataFrame(summary_rows).sort_values("ticker").reset_index(drop=True)
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    print("=" * 72)
    print("TASK 3A COMPLETE — leakage-free sequence preparation (no training)")
    print(f"Datasets -> {out_dir} | lookback={lookback} | 70/15/15 chronological")
    for _, r in summary.iterrows():
        print(f"  {r['ticker']}: train={r['n_train']} val={r['n_val']} "
              f"test={r['n_test']} feats={r['n_features']}")
    print("=" * 72)
    return {"output_dir": str(out_dir), "tickers": tickers,
            "n_tickers": len(tickers)}


# ===========================================================================
# Task 3B — Leakage-free deep-learning training (LSTM/GRU/BiLSTM/Transformer)
# ===========================================================================
# Neural-network training and evaluation ONLY. Datasets from Task 3A
# (``data/processed/deep_learning/``) are read, never regenerated or altered.
# Raw market data, ``data/processed/features/``, and Task 2C/2D outputs are
# only ever read, never written.
#
# Target: target_t = log(AdjClose_{t+1} / AdjClose_t), one scalar per
# 30-day x 30-feature input window. Test set is evaluated exactly once per
# ticker/model AFTER architecture + config are fixed, early stopping is
# complete, and the best validation-loss checkpoint is restored.

import random as _random

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - environment without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _exc

SEED = 42

TASK3B_CONFIG: dict = {
    "seed": SEED,
    "input_size": 30,
    "lookback": LOOKBACK,
    "hidden_size": 64,
    "recurrent_layers": 1,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "early_stopping_patience": 8,
    "weight_decay": 1e-5,
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "dim_feedforward": 128,
    },
}

MODEL_NAMES_3B = ["lstm", "gru", "bilstm", "transformer"]

DL_INPUT_DIR = Path("data/processed/deep_learning")
DL_TRAINING_DIR = Path("data/processed/deep_learning_training")
DL_CHECKPOINT_SUBDIR = "checkpoints"


def require_torch() -> None:
    """Raise a helpful error when PyTorch is unavailable."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Task 3B training but could not be "
            f"imported: {_TORCH_IMPORT_ERROR}. Install CPU/CUDA torch first."
        )


def set_seed(seed: int = SEED) -> None:
    """Fix Python/NumPy/PyTorch seeds; deterministic CUDA where practical."""
    require_torch()
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch without warn_only.
        torch.use_deterministic_algorithms(True)


def get_device() -> "torch.device":
    """CUDA when available, else CPU (CUDA verified on RTX 4060)."""
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report_device() -> dict:
    """GPU/CPU + memory info for logging before training (no training here)."""
    require_torch()
    cuda = bool(torch.cuda.is_available())
    info: dict = {
        "torch_version": str(torch.__version__),
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
    }
    if cuda:
        try:
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = str(props.name)
            info["device_count"] = int(torch.cuda.device_count())
            info["total_memory_bytes"] = int(props.total_memory)
            info["total_memory_gb"] = round(float(props.total_memory) / 1024**3, 2)
            info["capability"] = f"{props.major}.{props.minor}"
            info["multiprocessor_count"] = int(props.multi_processor_count)
        except Exception as exc:
            info["device_query_error"] = str(exc)
    return info


def estimate_workload(
    tickers: list[str] | None = None,
    max_epochs: int = TASK3B_CONFIG["max_epochs"],
    batch_size: int = TASK3B_CONFIG["batch_size"],
) -> dict:
    """Estimate training workload from Task 3A sequence counts (no training)."""
    if tickers is None:
        tickers = [sanitize_filename(t) for t in SUPERVISED_TICKERS]
    n_models = len(MODEL_NAMES_3B)
    total_train = 0
    per_ticker: dict = {}
    for t in tickers:
        path = DL_INPUT_DIR / f"{t}.npz"
        if path.exists():
            z = np.load(path)
            ntr = int(len(z["X_train"]))
            total_train += ntr
            per_ticker[t] = ntr
    batches = (total_train + batch_size - 1) // batch_size if total_train else 0
    return {
        "n_tickers": len(tickers),
        "n_models": n_models,
        "n_runs": len(tickers) * n_models,
        "total_train_sequences": total_train,
        "train_per_ticker": per_ticker,
        "batches_per_epoch_all_runs": batches * n_models,
        "max_steps_all_runs": batches * n_models * max_epochs,
        "note": "tiny models (~30x30 inputs); seconds per run on RTX 4060 CUDA",
    }


# ---------------------------------------------------------------------------
# Reusable PyTorch models (all: (batch, 30, 30) -> (batch,) scalar)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class LSTMRegressor(nn.Module):
        """1-2 layer LSTM + dropout + linear head -> scalar log-return."""

        def __init__(self, input_size: int = 30, hidden_size: int = 64,
                     num_layers: int = 1, dropout: float = 0.1) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            last = self.dropout(out[:, -1, :])
            return self.head(last).squeeze(-1)

    class GRURegressor(nn.Module):
        """1-2 layer GRU + dropout + linear head -> scalar log-return."""

        def __init__(self, input_size: int = 30, hidden_size: int = 64,
                     num_layers: int = 1, dropout: float = 0.1) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.gru(x)
            last = self.dropout(out[:, -1, :])
            return self.head(last).squeeze(-1)

    class BiLSTMRegressor(nn.Module):
        """Bidirectional LSTM over the historical 30-step window only.

        Bidirectionality operates strictly WITHIN the input window
        (t-29..t); no observation after prediction timestamp t is ever
        accessed. Backward pass runs from t down to t-29, never beyond t.
        """

        def __init__(self, input_size: int = 30, hidden_size: int = 64,
                     num_layers: int = 1, dropout: float = 0.1) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(2 * hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            last = self.dropout(out[:, -1, :])
            return self.head(last).squeeze(-1)

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding (batch_first)."""

        def __init__(self, d_model: int, max_len: int = 64,
                     dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.dropout(x + self.pe[:, :x.size(1), :])

    class TransformerRegressor(nn.Module):
        """Small encoder-only Transformer: 30 feats -> d_model -> scalar."""

        def __init__(self, input_size: int = 30, d_model: int = 64,
                     nhead: int = 4, num_layers: int = 2,
                     dropout: float = 0.1, dim_feedforward: int = 128) -> None:
            super().__init__()
            self.proj = nn.Linear(input_size, d_model)
            self.pos = PositionalEncoding(d_model, dropout=dropout)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.proj(x)
            h = self.pos(h)
            h = self.encoder(h)
            last = self.dropout(h[:, -1, :])
            return self.head(last).squeeze(-1)

    class SequenceDataset(Dataset):
        """Thin torch wrapper over Task 3A numpy arrays (read-only)."""

        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
            self.y = torch.as_tensor(np.asarray(y, dtype=np.float32))

        def __len__(self) -> int:
            return int(self.X.shape[0])

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx]

else:  # pragma: no cover - placeholders when torch missing

    class LSTMRegressor:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch unavailable")

    class GRURegressor:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch unavailable")

    class BiLSTMRegressor:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch unavailable")

    class TransformerRegressor:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch unavailable")

    class SequenceDataset:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch unavailable")


def build_model(model_name: str, config: dict | None = None):
    """Instantiate one Task 3B architecture from the shared config."""
    require_torch()
    cfg = dict(TASK3B_CONFIG if config is None else config)
    name = model_name.lower()
    if name == "lstm":
        return LSTMRegressor(cfg["input_size"], cfg["hidden_size"],
                             cfg["recurrent_layers"], cfg["dropout"])
    if name == "gru":
        return GRURegressor(cfg["input_size"], cfg["hidden_size"],
                            cfg["recurrent_layers"], cfg["dropout"])
    if name == "bilstm":
        return BiLSTMRegressor(cfg["input_size"], cfg["hidden_size"],
                               cfg["recurrent_layers"], cfg["dropout"])
    if name == "transformer":
        t = cfg["transformer"]
        return TransformerRegressor(cfg["input_size"], t["d_model"],
                                    t["nhead"], t["num_layers"],
                                    t["dropout"], t["dim_feedforward"])
    raise ValueError(f"unknown model {model_name!r}; expected {MODEL_NAMES_3B}")


def model_hyperparams(model_name: str, config: dict | None = None) -> dict:
    """Exact architecture settings for metadata (reproducibility)."""
    cfg = dict(TASK3B_CONFIG if config is None else config)
    name = model_name.lower()
    if name in ("lstm", "gru", "bilstm"):
        return {"input_size": cfg["input_size"], "hidden_size": cfg["hidden_size"],
                "num_layers": cfg["recurrent_layers"], "dropout": cfg["dropout"],
                "bidirectional": name == "bilstm"}
    t = cfg["transformer"]
    return {"input_size": cfg["input_size"], "d_model": t["d_model"],
            "nhead": t["nhead"], "num_layers": t["num_layers"],
            "dropout": t["dropout"], "dim_feedforward": t["dim_feedforward"]}


def compute_dl_metrics(y_true, y_pred) -> dict:
    """Test metrics using Task 2C definitions (RMSE/MAE primary)."""
    from src.models import compute_metrics
    return compute_metrics(np.asarray(y_true, dtype=float),
                           np.asarray(y_pred, dtype=float))


def load_dl_arrays(ticker: str,
                   input_dir: str | Path = DL_INPUT_DIR) -> dict:
    """Read-only load of Task 3A arrays for one ticker (never modifies)."""
    path = Path(input_dir) / f"{sanitize_filename(ticker)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Task 3A dataset missing for {ticker!r}: {path}. "
            "Run Task 3A preparation first; Task 3B never regenerates data.")
    z = np.load(path, allow_pickle=False)
    return {
        "X_train": np.asarray(z["X_train"], dtype=float),
        "y_train": np.asarray(z["y_train"], dtype=float),
        "X_val": np.asarray(z["X_val"], dtype=float),
        "y_val": np.asarray(z["y_val"], dtype=float),
        "X_test": np.asarray(z["X_test"], dtype=float),
        "y_test": np.asarray(z["y_test"], dtype=float),
        "test_dates": np.asarray(z["test_dates"]),
        "feature_names": [str(v) for v in np.asarray(z["feature_names"])],
    }


def train_one_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict | None = None,
    device: "torch.device | str | None" = None,
    seed: int = SEED,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Train on TRAIN ONLY; early-stop + checkpoint on VALIDATION loss only.

    Never touches test data. Returns the model (best-val checkpoint
    restored), per-epoch history, best epoch (1-indexed) and best val loss.
    """
    require_torch()
    cfg = dict(TASK3B_CONFIG if config is None else config)
    set_seed(seed)
    dev = torch.device(device) if device is not None else get_device()
    model = build_model(model_name, cfg).to(dev)
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]),
                              shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]),
                            shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                           weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.MSELoss()
    patience = int(cfg["early_stopping_patience"])
    max_epochs = int(cfg["max_epochs"])

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = tot / max(n, 1)
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = loss_fn(model(xb), yb)
                vtot += float(loss.item()) * len(xb)
                vn += len(xb)
        val_loss = vtot / max(vn, 1)
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                        "validation_loss": float(val_loss)})
        if val_loss < best_val - 1e-12:
            best_val = float(val_loss)
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                p = Path(checkpoint_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model_name, "epoch": epoch,
                            "validation_loss": float(val_loss),
                            "state_dict": best_state,
                            "config": {k: v for k, v in cfg.items()
                                       if k != "transformer"},
                            "seed": seed}, p)
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_validation_loss": float(best_val), "device": str(dev)}


@torch.no_grad() if _TORCH_AVAILABLE else (lambda f: f)
def predict_with_model(model, X: np.ndarray,
                       device: "torch.device | str | None" = None,
                       batch_size: int = 64) -> np.ndarray:
    """Deterministic inference for a restored checkpoint (eval mode)."""
    require_torch()
    dev = torch.device(device) if device is not None else get_device()
    model = model.to(dev).eval()
    loader = DataLoader(SequenceDataset(X, np.zeros(len(X))),
                        batch_size=batch_size, shuffle=False)
    outs: list[np.ndarray] = []
    for xb, _ in loader:
        outs.append(model(xb.to(dev)).detach().cpu().numpy())
    return np.concatenate(outs).astype(float)


def save_checkpoint(model, model_name: str, ticker: str, epoch: int,
                    val_loss: float, path: str | Path,
                    seed: int = SEED) -> None:
    """Persist best-validation checkpoint with ticker/model/epoch metadata."""
    require_torch()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ticker": ticker, "model": model_name, "epoch": int(epoch),
                "validation_loss": float(val_loss),
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
                "seed": seed}, p)


def load_checkpoint(model_name: str, path: str | Path,
                    config: dict | None = None,
                    device: "torch.device | str | None" = None):
    """Restore a saved best checkpoint (for deterministic re-evaluation)."""
    require_torch()
    cfg = dict(TASK3B_CONFIG if config is None else config)
    dev = torch.device(device) if device is not None else get_device()
    model = build_model(model_name, cfg).to(dev)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def train_ticker_models(
    ticker: str,
    config: dict | None = None,
    device: "torch.device | str | None" = None,
    seed: int = SEED,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Full leakage-free procedure for one ticker (test evaluated ONCE last).

    1. Load Task 3A arrays. 2. For each architecture: train on TRAIN ONLY,
    validate per-epoch, keep best-val checkpoint, early-stop, restore best.
    3. ONLY after finalization, evaluate each restored model once on TEST.
    """
    require_torch()
    cfg = dict(TASK3B_CONFIG if config is None else config)
    dev = torch.device(device) if device is not None else get_device()
    data = load_dl_arrays(ticker)
    metric_rows, hist_rows, pred_rows, best_rows = [], [], [], []
    failures: list[dict] = []
    for name in MODEL_NAMES_3B:
        ckpt_path = (Path(checkpoint_dir) / f"{sanitize_filename(ticker)}_{name}_best.pt"
                     if checkpoint_dir is not None else None)
        try:
            res = train_one_model(
                name, data["X_train"], data["y_train"],
                data["X_val"], data["y_val"], config=cfg, device=dev,
                seed=seed, checkpoint_path=ckpt_path)
        except Exception as exc:
            failures.append({"ticker": ticker, "model": name,
                             "error": str(exc)})
            continue
        for h in res["history"]:
            hist_rows.append({"ticker": ticker, "model": name, **h})
        # Finalize: checkpoint already holds best-val weights; re-save with
        # ticker metadata when a dir was given (same weights, richer header).
        if ckpt_path is not None:
            save_checkpoint(res["model"], name, ticker, res["best_epoch"],
                            res["best_validation_loss"], ckpt_path, seed=seed)
        # TEST ISOLATION: single evaluation of the finalized model.
        preds = predict_with_model(res["model"], data["X_test"], device=dev,
                                   batch_size=int(cfg["batch_size"]))
        metrics = compute_dl_metrics(data["y_test"], preds)
        metric_rows.append({"ticker": ticker, "model": name,
                            "rmse": metrics["rmse"], "mae": metrics["mae"],
                            "mape": metrics["mape"], "r2": metrics["r2"],
                            "directional_accuracy": metrics["directional_accuracy"],
                            "best_epoch": res["best_epoch"],
                            "best_validation_loss": res["best_validation_loss"],
                            "seed": seed})
        best_rows.append({"ticker": ticker, "model": name,
                          "best_epoch": res["best_epoch"],
                          "best_validation_loss": res["best_validation_loss"],
                          "seed": seed,
                          "checkpoint": str(ckpt_path) if ckpt_path else "",
                          "hyperparams": str(model_hyperparams(name, cfg))})
        for d, a, p in zip(data["test_dates"], data["y_test"], preds):
            pred_rows.append({"ticker": ticker, "model": name,
                              "date": pd.to_datetime(d),
                              "actual_return": float(a),
                              "predicted_return": float(p)})
    return {"ticker": ticker, "metrics": metric_rows, "history": hist_rows,
            "predictions": pred_rows, "best": best_rows, "failures": failures}


def run_training(
    tickers: list[str] | None = None,
    config: dict | None = None,
    output_dir: str | Path = DL_TRAINING_DIR,
    input_dir: str | Path = DL_INPUT_DIR,
    seed: int = SEED,
    device: "torch.device | str | None" = None,
) -> dict:
    """Train/evaluate all Task 3B models for all tickers (deterministic).

    Reads ``input_dir`` only; writes CSVs + config + checkpoints under
    ``output_dir``. Never touches raw/features/Task 2C/2D locations.
    """
    require_torch()
    import json as _json

    import pandas as _pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = dict(TASK3B_CONFIG if config is None else config)
    cfg["seed"] = seed
    dev_info = report_device()
    dev = torch.device(device) if device is not None else get_device()
    set_seed(seed)
    if tickers is None:
        tickers = [sanitize_filename(t) for t in SUPERVISED_TICKERS]
    tickers = [sanitize_filename(t) for t in tickers]
    global DL_INPUT_DIR
    _prev_input, DL_INPUT_DIR = DL_INPUT_DIR, Path(input_dir)
    try:
        workload = estimate_workload(tickers, cfg["max_epochs"], cfg["batch_size"])
    finally:
        DL_INPUT_DIR = _prev_input
    logger.info("device=%s (%s) torch=%s cuda=%s", dev, dev_info.get("device_name", "CPU"),
                dev_info.get("torch_version"), dev_info.get("cuda_available"))
    logger.info("workload: %d tickers x %d models = %d runs, %d train seqs",
                workload["n_tickers"], workload["n_models"], workload["n_runs"],
                workload["total_train_sequences"])

    out_dir = Path(output_dir)
    ckpt_dir = out_dir / DL_CHECKPOINT_SUBDIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_m, all_h, all_p, all_b, all_f = [], [], [], [], []
    for ticker in tickers:
        r = train_ticker_models(ticker, config=cfg, device=dev, seed=seed,
                                checkpoint_dir=ckpt_dir)
        all_m += r["metrics"]
        all_h += r["history"]
        all_p += r["predictions"]
        all_b += r["best"]
        all_f += r["failures"]
        logger.info("%s: %d models ok, %d failed", ticker,
                    len(r["metrics"]), len(r["failures"]))

    metrics_df = _pd.DataFrame(all_m)
    if len(metrics_df):
        metrics_df = metrics_df.sort_values(["ticker", "model"]).reset_index(drop=True)
        metrics_df.to_csv(out_dir / "dl_metrics.csv", index=False)
    _pd.DataFrame(all_h).sort_values(["ticker", "model", "epoch"]).to_csv(
        out_dir / "training_history.csv", index=False)
    _pd.DataFrame(all_p).sort_values(["ticker", "model", "date"]).to_csv(
        out_dir / "dl_predictions.csv", index=False)
    _pd.DataFrame(all_b).sort_values(["ticker", "model"]).to_csv(
        out_dir / "dl_best_models.csv", index=False)
    meta = {
        "task": "3B leakage-free deep-learning training",
        "target": "log(AdjClose_{t+1} / AdjClose_t)",
        "input_shape": [LOOKBACK, cfg["input_size"]],
        "output": "one scalar predicted next-day log return",
        "device": dev_info,
        "device_used": str(dev),
        "torch_version": dev_info.get("torch_version"),
        "seed": seed,
        "architecture": {m: model_hyperparams(m, cfg) for m in MODEL_NAMES_3B},
        "training": {
            "optimizer": "Adam",
            "loss": "MSELoss",
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "max_epochs": cfg["max_epochs"],
            "early_stopping_patience": cfg["early_stopping_patience"],
            "weight_decay": cfg["weight_decay"],
            "model_selection": "validation loss only; best checkpoint restored",
            "test_isolation": ("test evaluated once after checkpoint selection; "
                               "never used for tuning/early stopping/architecture"),
            "refit_policy": "no train+val refit; controlled test comparison",
        },
        "metrics_note": ("RMSE/MAE primary; directional_accuracy separate; "
                         "definitions identical to Task 2C compute_metrics; no profitability claims"),
        "bilstm_note": ("bidirectionality operates only within the historical "
                        "30-step window, never on observations after t"),
        "workload": workload,
        "failures": all_f,
    }
    (out_dir / "config.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 72)
    print("TASK 3B COMPLETE — leakage-free DL training (test evaluated once)")
    print(f"Device -> {dev} ({dev_info.get('device_name', 'CPU')}) | "
          f"torch={dev_info.get('torch_version')} | seed={seed}")
    print(f"Outputs -> {out_dir} | runs={workload['n_runs']} "
          f"ok={len(all_m)} failed={len(all_f)}")
    print("=" * 72)
    return {"output_dir": str(out_dir), "device": dev_info,
            "n_metrics": len(all_m), "failures": all_f, "seed": seed}


def main(config_path: str | Path = "config.yaml", *args) -> dict:
    """CLI: ``python -m src.deep_learning`` trains Task 3B (CUDA if present).

    ``python -m src.deep_learning --prepare`` re-runs Task 3A preparation.
    """
    import sys as _sys

    argv = list(args) or _sys.argv[1:]
    if "--prepare" in argv:
        return prepare_all_tickers(config_path=config_path)
    return run_training()


if __name__ == "__main__":
    main()
