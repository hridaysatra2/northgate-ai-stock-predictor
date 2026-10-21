"""Tests for Task 3B leakage-free DL training (src.deep_learning).

Behavioral assertions on real training/evaluation, not mere existence
checks. Heavy production outputs are verified from saved files; live
training uses tiny synthetic data on CPU for speed and determinism.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import deep_learning as dl3b
from src.models import compute_metrics as classical_compute_metrics

OUT_DIR = Path("data/processed/deep_learning_training")
RAW_DIR = Path("data/raw")
FEATURES_DIR = Path("data/processed/features")
MODELS_DIR = Path("data/processed/models")
EVAL_DIR = Path("data/processed/evaluation")

TINY_CFG = {
    "seed": 42,
    "input_size": 30,
    "lookback": 30,
    "hidden_size": 8,
    "recurrent_layers": 1,
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "batch_size": 16,
    "max_epochs": 4,
    "early_stopping_patience": 8,
    "weight_decay": 0.0,
    "transformer": {"d_model": 8, "nhead": 2, "num_layers": 1,
                    "dropout": 0.0, "dim_feedforward": 16},
}


def make_tiny_arrays(n_train=64, n_val=24, n_test=20, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "X_train": rng.normal(0, 1, (n_train, 30, 30)),
        "y_train": rng.normal(0, 0.01, n_train),
        "X_val": rng.normal(0, 1, (n_val, 30, 30)),
        "y_val": rng.normal(0, 0.01, n_val),
        "X_test": rng.normal(0, 1, (n_test, 30, 30)),
        "y_test": rng.normal(0, 0.01, n_test),
    }


def _cpu(name, **kw):
    dl3b.set_seed(42)
    m = dl3b.build_model(name, TINY_CFG, **kw) if False else dl3b.build_model(name, TINY_CFG)
    return m.eval()


# 1. model output shape is one scalar per sequence.
@pytest.mark.parametrize("arch", ["lstm", "gru", "bilstm", "transformer"])
def test_1_output_is_scalar_per_sequence(arch):
    dl3b.set_seed(42)
    model = dl3b.build_model(arch, TINY_CFG).eval()
    x = torch.randn(7, 30, 30)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (7,), f"{arch} output shape {tuple(out.shape)} != (7,)"
    assert torch.isfinite(out).all()


# 2-5. each architecture accepts (batch,30,30).
@pytest.mark.parametrize("arch", ["lstm", "gru", "bilstm", "transformer"])
def test_2_5_accepts_30x30(arch):
    dl3b.set_seed(0)
    model = dl3b.build_model(arch, TINY_CFG).eval()
    for b in (1, 5):
        x = torch.randn(b, 30, 30)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b,)
    # Wrong feature dim must fail loudly, not silently broadcast.
    with pytest.raises(Exception):
        with torch.no_grad():
            model(torch.randn(2, 30, 7))


# 6. deterministic initialization/training with fixed seed (CPU).
def test_6_deterministic_init_and_training():
    dl3b.set_seed(123)
    a = dl3b.build_model("lstm", TINY_CFG)
    dl3b.set_seed(123)
    b = dl3b.build_model("lstm", TINY_CFG)
    for pa, pb in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(pa.detach().numpy(), pb.detach().numpy())
    d = make_tiny_arrays()
    r1 = dl3b.train_one_model("gru", d["X_train"], d["y_train"],
                              d["X_val"], d["y_val"], config=TINY_CFG,
                              device="cpu", seed=42)
    r2 = dl3b.train_one_model("gru", d["X_train"], d["y_train"],
                              d["X_val"], d["y_val"], config=TINY_CFG,
                              device="cpu", seed=42)
    assert [h["train_loss"] for h in r1["history"]] == pytest.approx(
        [h["train_loss"] for h in r2["history"]])
    assert r1["best_epoch"] == r2["best_epoch"]
    assert r1["best_validation_loss"] == pytest.approx(r2["best_validation_loss"])


# 7. training uses only train arrays.
def test_7_training_uses_only_train():
    d = make_tiny_arrays()
    base = dl3b.train_one_model("lstm", d["X_train"], d["y_train"],
                                d["X_val"], d["y_val"], config=TINY_CFG,
                                device="cpu", seed=42)
    # Corrupt val/test drastically: train-loss trajectory must be identical.
    shocked_val = d["X_val"] * 999.0 + 50.0
    shocked_y = d["y_val"] * -777.0
    shocked = dl3b.train_one_model("lstm", d["X_train"], d["y_train"],
                                   shocked_val, shocked_y, config=TINY_CFG,
                                   device="cpu", seed=42)
    assert [h["train_loss"] for h in shocked["history"]] == pytest.approx(
        [h["train_loss"] for h in base["history"]])
    # ...while validation losses do change (val is actually read).
    assert [h["validation_loss"] for h in shocked["history"]] != pytest.approx(
        [h["validation_loss"] for h in base["history"]])


# 8. validation is used for early stopping.
def test_8_validation_drives_early_stopping():
    d = make_tiny_arrays()
    cfg = dict(TINY_CFG, max_epochs=30, early_stopping_patience=3)
    res = dl3b.train_one_model("gru", d["X_train"], d["y_train"],
                               d["X_val"], d["y_val"], config=cfg,
                               device="cpu", seed=42)
    hist = res["history"]
    assert len(hist) >= 2
    vals = [h["validation_loss"] for h in hist]
    assert res["best_validation_loss"] == pytest.approx(min(vals))
    assert hist[res["best_epoch"] - 1]["validation_loss"] == pytest.approx(min(vals))
    # Early stopped: stopped patience epochs after best (unless max reached).
    assert len(hist) == min(cfg["max_epochs"], res["best_epoch"] + cfg["early_stopping_patience"])
    # More patience never stops earlier.
    cfg2 = dict(cfg, early_stopping_patience=30)
    res2 = dl3b.train_one_model("gru", d["X_train"], d["y_train"],
                                d["X_val"], d["y_val"], config=cfg2,
                                device="cpu", seed=42)
    assert len(res2["history"]) >= len(hist)


# 9. test evaluation occurs only after checkpoint selection.
def test_9_test_only_after_checkpoint(tmp_path):
    d = make_tiny_arrays()
    ckpt = tmp_path / "gru_best.pt"
    res = dl3b.train_one_model("gru", d["X_train"], d["y_train"],
                               d["X_val"], d["y_val"], config=TINY_CFG,
                               device="cpu", seed=42, checkpoint_path=ckpt)
    assert ckpt.exists()
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["epoch"] == res["best_epoch"]
    # Test arrays have no effect on the checkpoint: retrain with shocked
    # test data is impossible by signature (no test arg exists); instead
    # verify predictions match test length and checkpoint predates them.
    preds = dl3b.predict_with_model(res["model"], d["X_test"], device="cpu")
    assert len(preds) == len(d["y_test"])
    # Production: predictions align with Task 3A test dates exactly.
    z = np.load(dl3b.DL_INPUT_DIR / "AAPL.npz")
    prod = pd.read_csv(OUT_DIR / "dl_predictions.csv")
    sub = prod[(prod.ticker == "AAPL") & (prod.model == "gru")].sort_values("date")
    assert len(sub) == len(z["y_test"])
    np.testing.assert_array_equal(
        pd.to_datetime(sub["date"]).to_numpy().astype("datetime64[ns]"),
        np.asarray(z["test_dates"]).astype("datetime64[ns]"))


# 10. metrics match established Task 2C/2D definitions.
def test_10_metrics_match_classical_definitions():
    rng = np.random.default_rng(7)
    y_true = rng.normal(0, 0.01, 200)
    y_pred = rng.normal(0, 0.012, 200)
    got = dl3b.compute_dl_metrics(y_true, y_pred)
    want = classical_compute_metrics(y_true, y_pred)
    assert set(got) == set(want)
    for k in want:
        assert got[k] == pytest.approx(want[k], nan_ok=True)
    # Direction convention: sign agreement, exact zeros safe.
    assert dl3b.compute_dl_metrics(np.array([0.0, 0.01, -0.02]),
                                   np.array([0.0, -0.01, -0.03]))["directional_accuracy"] == pytest.approx(2 / 3)


# 11. predictions and metrics have expected schema.
def test_11_output_schema():
    metrics = pd.read_csv(OUT_DIR / "dl_metrics.csv")
    preds = pd.read_csv(OUT_DIR / "dl_predictions.csv")
    hist = pd.read_csv(OUT_DIR / "training_history.csv")
    best = pd.read_csv(OUT_DIR / "dl_best_models.csv")
    assert set(metrics.columns) >= {"ticker", "model", "rmse", "mae", "mape",
                                    "r2", "directional_accuracy", "best_epoch",
                                    "best_validation_loss", "seed"}
    assert set(preds.columns) >= {"ticker", "model", "date", "actual_return",
                                  "predicted_return"}
    assert set(hist.columns) >= {"ticker", "model", "epoch", "train_loss",
                                 "validation_loss"}
    assert set(best.columns) >= {"ticker", "model", "best_epoch",
                                 "best_validation_loss", "checkpoint"}
    assert set(metrics["model"]) == {"lstm", "gru", "bilstm", "transformer"}
    assert metrics["ticker"].nunique() == 11
    assert len(metrics) == 44
    # Every metric row has matching prediction + history + best rows.
    for _, r in metrics.iterrows():
        assert len(preds[(preds.ticker == r["ticker"]) & (preds.model == r["model"])]) > 0
        assert len(hist[(hist.ticker == r["ticker"]) & (hist.model == r["model"])]) > 0


# 12. no raw/features/Task 2C/Task 2D files are modified.
def _hash_files(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}


def test_12_protected_outputs_untouched(tmp_path):
    before = {}
    for label, paths in (
        ("raw", list(RAW_DIR.rglob("*.parquet"))),
        ("feat", list(FEATURES_DIR.glob("*.parquet"))),
        ("2c", list(MODELS_DIR.glob("*.csv"))),
        ("2d", list(EVAL_DIR.glob("*.csv")) + list(EVAL_DIR.glob("*.txt"))),
    ):
        assert len(paths) > 0, f"no files found for {label}"
        before[label] = _hash_files(paths)
    d = make_tiny_arrays(n_train=48, n_val=16, n_test=10)
    dl3b.train_one_model("lstm", d["X_train"], d["y_train"], d["X_val"],
                         d["y_val"], config=TINY_CFG, device="cpu", seed=1,
                         checkpoint_path=tmp_path / "x.pt")
    dl3b.train_ticker_models("KO", config=TINY_CFG, device="cpu", seed=1,
                             checkpoint_dir=tmp_path / "ckpt")
    after = {label: _hash_files(
        list(RAW_DIR.rglob("*.parquet")) if label == "raw"
        else list(FEATURES_DIR.glob("*.parquet")) if label == "feat"
        else list(MODELS_DIR.glob("*.csv")) if label == "2c"
        else list(EVAL_DIR.glob("*.csv")) + list(EVAL_DIR.glob("*.txt")))
        for label in before}
    assert before == after


# 13. saved training history is non-empty and ordered by epoch.
def test_13_history_nonempty_ordered():
    hist = pd.read_csv(OUT_DIR / "training_history.csv")
    assert len(hist) > 0
    for (t, m), g in hist.groupby(["ticker", "model"]):
        ep = g.sort_values("epoch")["epoch"].to_numpy()
        assert ep[0] == 1
        assert (np.diff(ep) == 1).all(), f"{t}/{m} epochs not 1..N ordered"
        assert np.isfinite(g["train_loss"].to_numpy()).all()
        assert np.isfinite(g["validation_loss"].to_numpy()).all()
        assert (g["train_loss"].to_numpy() >= 0).all()
        assert (g["validation_loss"].to_numpy() >= 0).all()


# 14. saved best checkpoint metadata identifies model/ticker/epoch/val loss.
def test_14_checkpoint_metadata():
    ckpts = sorted((OUT_DIR / "checkpoints").glob("*_best.pt"))
    assert len(ckpts) == 44
    metrics = pd.read_csv(OUT_DIR / "dl_metrics.csv")
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert {"model", "epoch", "validation_loss", "state_dict"} <= set(ck)
        stem = p.stem  # e.g. AAPL_bilstm_best
        ticker, model = stem[: -len("_best")].rsplit("_", 1)
        row = metrics[(metrics.ticker == ticker) & (metrics.model == model)]
        assert len(row) == 1
        assert ck["epoch"] == int(row.iloc[0]["best_epoch"])
        assert ck["validation_loss"] == pytest.approx(float(row.iloc[0]["best_validation_loss"]))
        assert int(row.iloc[0]["seed"]) == 42


# 15. no NaN/Inf in predictions or primary metrics.
def test_15_no_nan_inf():
    metrics = pd.read_csv(OUT_DIR / "dl_metrics.csv")
    preds = pd.read_csv(OUT_DIR / "dl_predictions.csv")
    for c in ("rmse", "mae", "r2", "directional_accuracy"):
        assert np.isfinite(metrics[c].to_numpy()).all(), f"non-finite {c}"
    assert np.isfinite(preds["predicted_return"].to_numpy()).all()
    assert np.isfinite(preds["actual_return"].to_numpy()).all()
    # MAPE may be NaN only if all actuals were exactly zero (never here).
    assert preds["actual_return"].abs().sum() > 0


# 16. repeated evaluation of the same saved checkpoint is deterministic.
def test_16_checkpoint_reeval_deterministic():
    ckpt_path = OUT_DIR / "checkpoints" / "AAPL_lstm_best.pt"
    data = dl3b.load_dl_arrays("AAPL")
    m1, _ = dl3b.load_checkpoint("lstm", ckpt_path, device="cpu")
    m2, _ = dl3b.load_checkpoint("lstm", ckpt_path, device="cpu")
    p1 = dl3b.predict_with_model(m1, data["X_test"], device="cpu")
    p2 = dl3b.predict_with_model(m2, data["X_test"], device="cpu")
    np.testing.assert_array_equal(p1, p2)
    # And matches the stored production (CUDA-trained) predictions closely;
    # CPU re-evaluation of CUDA-trained weights differs only by float noise.
    prod = pd.read_csv(OUT_DIR / "dl_predictions.csv")
    sub = prod[(prod.ticker == "AAPL") & (prod.model == "lstm")].sort_values("date")
    np.testing.assert_allclose(sub["predicted_return"].to_numpy(), p1,
                               rtol=1e-3, atol=1e-4)
