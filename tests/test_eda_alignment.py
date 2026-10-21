"""Regression tests for EDA date alignment (F01).

Synthetic series with different trading dates; verifies observations are
joined on actual dates, not positions. No I/O, no network.
"""

import numpy as np
import pandas as pd
import pytest

from src.eda import align_series_by_date, correlation_matrix


def test_alignment_uses_dates_not_positions():
    # Overlapping dates: 2020-01-02, 03, 06. Each ticker also has one
    # unique date that must be excluded, not positionally paired.
    dates_a = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
    )
    dates_b = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    )
    # Values on overlapping dates are perfectly correlated (B = 2*A).
    # A values: [0.99 (unmatched), 0.01, -0.02, 0.03]
    # B values: [0.02, -0.04, 0.06 (matched), -0.99 (unmatched)]
    vals_a = pd.Series([0.99, 0.01, -0.02, 0.03])
    vals_b = pd.Series([0.02, -0.04, 0.06, -0.99])
    wide = align_series_by_date(
        {"A": vals_a, "B": vals_b}, {"A": dates_a, "B": dates_b}
    )
    # Only genuinely matched dates survive.
    assert list(wide.index.date.astype(str)) == [
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
    ]
    assert len(wide) == 3
    # On matched dates the series are exactly proportional.
    corr = correlation_matrix(wide)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    # Positional pairing would have paired 0.99 with 0.02 etc. and would
    # not yield perfect correlation; guard explicitly.
    assert wide.loc[pd.Timestamp("2020-01-02"), "A"] == pytest.approx(0.01)
    assert wide.loc[pd.Timestamp("2020-01-02"), "B"] == pytest.approx(0.02)


def test_alignment_same_dates_expected_correlation():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    a = pd.Series([0.01, -0.02, 0.03])
    wide = align_series_by_date(
        {"A": a, "B": a * 2.0, "C": -a},
        {"A": dates, "B": dates, "C": dates},
    )
    assert len(wide) == 3
    corr = correlation_matrix(wide)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert corr.loc["A", "C"] == pytest.approx(-1.0)


def test_alignment_disjoint_dates_gives_empty_no_fill():
    wide = align_series_by_date(
        {"A": pd.Series([0.01, 0.02]), "B": pd.Series([0.03, 0.04])},
        {
            "A": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "B": pd.to_datetime(["2020-01-03", "2020-01-06"]),
        },
    )
    assert wide.empty
    assert wide.index.name == "date"


def test_alignment_rejects_duplicate_dates():
    with pytest.raises(ValueError, match="duplicate dates"):
        align_series_by_date(
            {"A": pd.Series([0.1, 0.2])},
            {"A": pd.to_datetime(["2020-01-01", "2020-01-01"])},
        )
