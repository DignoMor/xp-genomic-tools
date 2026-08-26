"""SPEC025 contract tests: RGTools.TSSRelativeCoordinates public API."""

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


def test_minus_strand_point_conversion_at_tss():
    """Minus-strand +1 maps to the genomic TSS base when window size is 1."""
    assert (
        tss_relative_to_track_index(
            strand="-",
            coord=1,
            start=100,
            end=110,
            tss=105,
            track_window_size=1,
        )
        == 5
    )


def test_minus_strand_point_conversion_upstream_and_downstream():
    """Minus reverses direction: negative coords are transcription-upstream."""
    # genomic_right = tss - _to_linear(coord); index = genomic - start for W=1
    # Issue #8: coord=-2 at tss=110 → genomic 112 → index 12 (not 8).
    assert (
        tss_relative_to_track_index(
            strand="-", coord=-2, start=100, end=120, tss=110, track_window_size=1
        )
        == 12
    )
    assert (
        tss_relative_to_track_index(
            strand="-", coord=-2, start=100, end=110, tss=105, track_window_size=1
        )
        == 7
    )  # 105 - (-2) = 107 → index 7
    assert (
        tss_relative_to_track_index(
            strand="-", coord=3, start=100, end=110, tss=105, track_window_size=1
        )
        == 3
    )  # 105 - (3 - 1) = 103 → index 3


def test_plus_motif_window_index_is_genomic_left():
    """Plus track_window_size>1 indexes the genomic-left 5-prime base."""
    # [100,120), tss=110, coord=+1, W=3 → genomic_left=110 → index 10
    assert (
        tss_relative_to_track_index(
            strand="+",
            coord=1,
            start=100,
            end=120,
            tss=110,
            track_window_size=3,
        )
        == 10
    )


def test_minus_motif_window_index_subtracts_padding():
    """Minus track_window_size>1 indexes genomic-right 5-prime via W-1 padding."""
    # [100,120), revTSS=110, coord=+1, W=3 → genomic_right=110 → index 8
    assert (
        tss_relative_to_track_index(
            strand="-",
            coord=1,
            start=100,
            end=120,
            tss=110,
            track_window_size=3,
        )
        == 8
    )


def test_minus_motif_window_upstream_and_downstream():
    """Minus W>1 reverses direction then subtracts W-1 for genomic-left index."""
    # [100,120), tss=110, W=3: genomic_right = tss - linear; index = genomic - start - 2
    assert (
        tss_relative_to_track_index(
            strand="-", coord=-2, start=100, end=120, tss=110, track_window_size=3
        )
        == 10
    )  # genomic_right 112 → index 10
    assert (
        tss_relative_to_track_index(
            strand="-", coord=3, start=100, end=120, tss=110, track_window_size=3
        )
        == 6
    )  # genomic_right 108 → index 6


def test_edge_positions_plus_and_minus_with_window():
    """Exact edge indices remain valid when the scored window fits."""
    # Plus: genomic_left at start, W=2 → index 0; window [start, start+2)
    assert (
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=100, track_window_size=2
        )
        == 0
    )
    # Minus: genomic_right at end-1, W=2 → index = (end-1)-start-(2-1) = length-2
    assert (
        tss_relative_to_track_index(
            strand="-", coord=1, start=100, end=110, tss=109, track_window_size=2
        )
        == 8
    )


def test_conversion_rejects_zero_coord():
    with pytest.raises(ValueError, match="[Zz]ero"):
        tss_relative_to_track_index(
            strand="+", coord=0, start=100, end=110, tss=105, track_window_size=1
        )


def test_conversion_rejects_out_of_bounds_index():
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=99, track_window_size=1
        )


def test_conversion_rejects_window_that_does_not_fit():
    """A scored window that would extend past end is out of bounds."""
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        # Plus W=3 at genomic_left=end-1 cannot fit
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=109, track_window_size=3
        )


def test_conversion_rejects_invalid_interval():
    with pytest.raises(ValueError, match="[Ii]nterval|[Ee]nd|[Ss]tart"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=110, end=100, tss=105, track_window_size=1
        )


def test_conversion_rejects_invalid_strand():
    with pytest.raises(ValueError, match="[Ss]trand"):
        tss_relative_to_track_index(
            strand="*", coord=1, start=100, end=110, tss=105, track_window_size=1
        )


def test_conversion_rejects_nonpositive_window_size():
    with pytest.raises(ValueError, match="track_window_size"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=105, track_window_size=0
        )


@pytest.mark.parametrize(
    "target,relaxation,expected",
    [
        (3, 0, [3]),
        (-1, 1, [-2, -1, 1]),
        (1, 1, [-1, 1, 2]),
        (-2, 2, [-4, -3, -2, -1, 1]),
        (2, 2, [-1, 1, 2, 3, 4]),
    ],
)
def test_iter_relaxed_window_cardinality_and_ascending_order(
    target, relaxation, expected
):
    """Relaxation r yields exactly 2r+1 ascending no-zero coordinates."""
    got = list(iter_relaxed_window(target, relaxation))
    assert got == expected
    assert len(got) == 2 * relaxation + 1
    assert got == sorted(got)
    assert 0 not in got


def test_iter_relaxed_window_rejects_zero_target_and_negative_relaxation():
    with pytest.raises(ValueError, match="[Zz]ero"):
        list(iter_relaxed_window(0, 0))
    with pytest.raises(ValueError, match="relaxation"):
        list(iter_relaxed_window(1, -1))


def test_conversion_rejects_both_interval_edge_overflows():
    """Indices before start and past the last fitting window are out of bounds."""
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=99, track_window_size=1
        )
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        tss_relative_to_track_index(
            strand="+", coord=1, start=100, end=110, tss=110, track_window_size=1
        )


def test_plus_and_minus_short_region_out_of_bounds():
    """Short regions reject coordinates whose scored windows cannot fit."""
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        tss_relative_to_track_index(
            strand="+", coord=5, start=100, end=103, tss=101, track_window_size=1
        )
    with pytest.raises(ValueError, match="[Oo]ut|[Bb]ound|[Ii]ndex|[Ii]nterval"):
        tss_relative_to_track_index(
            strand="-", coord=5, start=100, end=103, tss=101, track_window_size=1
        )
