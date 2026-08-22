"""SPEC025 contract tests: RGTools.TSSRelativeCoordinates public API (ticket 01)."""

from __future__ import annotations

import pytest

from RGTools.TSSRelativeCoordinates import (
    iter_relaxed_window,
    offset_tss_relative_coordinate,
    tss_relative_to_track_index,
)


def test_offset_skips_zero_when_crossing_tss():
    """Offsetting across the TSS skips coordinate zero (no-zero system)."""
    assert offset_tss_relative_coordinate(-1, 1) == 1
    assert offset_tss_relative_coordinate(1, -1) == -1
    assert offset_tss_relative_coordinate(-2, 3) == 2
    assert offset_tss_relative_coordinate(3, -2) == 1


def test_offset_identity_and_same_side():
    """Zero delta and same-side offsets preserve no-zero arithmetic."""
    assert offset_tss_relative_coordinate(5, 0) == 5
    assert offset_tss_relative_coordinate(-3, 0) == -3
    assert offset_tss_relative_coordinate(2, 3) == 5
    assert offset_tss_relative_coordinate(-5, -1) == -6


def test_offset_rejects_zero_coordinate():
    """Coordinate zero is invalid input."""
    with pytest.raises(ValueError, match="[Zz]ero"):
        offset_tss_relative_coordinate(0, 1)


def test_plus_strand_point_conversion_at_tss():
    """Plus-strand +1 maps to the genomic TSS base (track_window_size=1)."""
    # region [100, 110), tss=105 → +1 → genomic 105 → index 5
    assert (
        tss_relative_to_track_index(
            strand="+",
            coord=1,
            start=100,
            end=110,
            tss=105,
            track_window_size=1,
        )
        == 5
    )


def test_plus_strand_point_conversion_upstream_and_downstream():
    """Negative coords are upstream; positive >1 are downstream of TSS."""
    # genomic = tss + coord - 1 if coord > 0 else tss + coord
    assert (
        tss_relative_to_track_index(
            strand="+", coord=-2, start=100, end=110, tss=105, track_window_size=1
        )
        == 3
    )  # 105 + (-2) = 103 → index 3
    assert (
        tss_relative_to_track_index(
            strand="+", coord=3, start=100, end=110, tss=105, track_window_size=1
        )
        == 7
    )  # 105 + 3 - 1 = 107 → index 7


def test_plus_conversion_rejects_zero_coord():
    with pytest.raises(ValueError, match="[Zz]ero"):
        tss_relative_to_track_index(
            strand="+", coord=0, start=100, end=110, tss=105, track_window_size=1
        )


def test_plus_conversion_rejects_out_of_bounds_index():
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=99, track_window_size=1
        )


def test_plus_conversion_rejects_invalid_interval():
    with pytest.raises(ValueError, match="[Ii]nterval|[Ee]nd|[Ss]tart"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=110, end=100, tss=105, track_window_size=1
        )


def test_minus_strand_not_yet_delivered():
    with pytest.raises(ValueError, match="[Nn]ot yet|[Uu]nsupported"):
        tss_relative_to_track_index(
            strand="-", coord=1, start=100, end=110, tss=105, track_window_size=1
        )


def test_track_window_size_gt_one_not_yet_delivered():
    with pytest.raises(ValueError, match="[Nn]ot yet|[Uu]nsupported"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=105, track_window_size=2
        )


def test_iter_relaxed_window_exact_only():
    """Ticket 01 delivers relaxation=0; nonzero relaxation is rejected."""
    assert list(iter_relaxed_window(3, 0)) == [3]
    with pytest.raises(ValueError, match="[Nn]ot yet|[Uu]nsupported"):
        list(iter_relaxed_window(3, 1))
