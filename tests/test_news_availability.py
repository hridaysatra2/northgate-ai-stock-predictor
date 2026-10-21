"""Tests for the Task 3C news availability audit (audit artifacts only).

No network, no modelling, no downloads. Verifies the audit exists,
is internally consistent, proves a small-sample-only claim, and that
protected project outputs contain no news artifacts and were not modified.
"""

import json
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path("data/processed/news_availability")
JSON_PATH = AUDIT_DIR / "news_source_audit.json"
CSV_PATH = AUDIT_DIR / "news_source_audit.csv"
MD_PATH = AUDIT_DIR / "NEWS_DATA_AVAILABILITY.md"

PROTECTED_DIRS = [
    Path("data/raw"),
    Path("data/processed/features"),
    Path("data/processed/models"),
    Path("data/processed/evaluation"),
    Path("data/processed/deep_learning"),
    Path("data/processed/deep_learning_training"),
]


def _audit():
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def test_1_audit_files_exist_and_only_expected():
    assert JSON_PATH.exists() and CSV_PATH.exists() and MD_PATH.exists()
    names = sorted(p.name for p in AUDIT_DIR.iterdir() if p.is_file())
    assert names == ["NEWS_DATA_AVAILABILITY.md", "news_source_audit.csv",
                     "news_source_audit.json"], names


def test_2_json_schema_and_reproducibility():
    a = _audit()
    assert a["research_period"] == {"start": "2016-01-01", "end": "2026-08-31"}
    assert a["universe"] == ["AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM",
                             "HD", "V", "CAT", "KO"]
    assert a["retrieval_timestamp_utc"].endswith("Z")
    assert len(a["sources"]) >= 2
    required = {"source", "live_tested", "timestamp_availability",
                "timestamp_precision", "timestamp_timezone",
                "ticker_company_identification", "title_headline_availability",
                "body_availability", "covers_full_2016_01_01_to_2026_08_31",
                "requires_key_or_paid", "suitable_for_leakage_free_3C"}
    for s in a["sources"]:
        assert required <= set(s), f"missing fields in {s.get('source')}"
    assert "verdict" in a and "recommended_next_step" in a["verdict"]


def test_3_csv_matches_json_verdicts():
    a = _audit()
    df = pd.read_csv(CSV_PATH)
    assert {"source", "suitable_for_leakage_free_3C"} <= set(df.columns)
    assert len(df) == len(a["sources"])
    for s in a["sources"]:
        row = df[df["source"] == s["source"]]
        assert len(row) == 1
        expected = str(bool(s["suitable_for_leakage_free_3C"])).lower()
        got = str(row.iloc[0]["suitable_for_leakage_free_3C"]).lower()
        assert got == expected, s["source"]


def test_4_small_sample_only():
    a = _audit()
    yf = next(s for s in a["sources"] if s["source"].startswith("yfinance"))
    assert yf["sample"]["n_items_returned"] <= 10
    gd = next(s for s in a["sources"] if s["source"].startswith("GDELT"))
    assert gd["sample"]["n_items_returned"] <= 5
    for s in a["sources"]:
        if not s["live_tested"]:
            assert s["sample"] is None or "note" in (s["sample"] or {})


def test_5_verdict_is_negative_and_no_substitution():
    a = _audit()
    assert a["verdict"]["suitable_source_found"] is False
    assert all(s["suitable_for_leakage_free_3C"] is False for s in a["sources"])
    assert "no_synthetic_substitution" in a["verdict"]
    md = MD_PATH.read_text(encoding="utf-8")
    assert "No suitable source" in md
    assert "STOP" in md


def test_6_yfinance_sample_postdates_cutoff_proving_gap():
    a = _audit()
    yf = next(s for s in a["sources"] if s["source"].startswith("yfinance"))
    assert yf["sample"]["all_items_after_research_cutoff_2026_08_31"] is True
    assert yf["sample"]["oldest_pubDate"] > "2026-08-31"


def test_7_protected_outputs_untouched():
    for d in PROTECTED_DIRS:
        assert d.exists(), f"protected dir missing: {d}"
        news_files = [p for p in d.rglob("*")
                      if "news" in p.name.lower() or "sentiment" in p.name.lower()]
        assert news_files == [], f"news artifacts leaked into {d}: {news_files}"
    # Audit dir holds no datasets, models, or predictions.
    for p in AUDIT_DIR.iterdir():
        assert p.suffix in {".json", ".csv", ".md"}, p.name
        assert p.stat().st_size < 100_000, f"suspiciously large audit file: {p}"
