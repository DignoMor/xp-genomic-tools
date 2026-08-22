"""SPEC026 contract tests: GenomicElementTools select_tss_relative_track."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from GenomicElementTools.cli import GenomicElementTools
from RGTools.GenomicElements import GenomicElements

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tss_relative"
TINY_TREBED = FIXTURES / "tiny.trebed"
MOTIF_TREBED = FIXTURES / "motif_window.trebed"
TINY_BED3 = Path(__file__).resolve().parents[1] / "fixtures" / "get" / "tiny.bed3"


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _build_track(tmp_path: Path, values: np.ndarray) -> Path:
    path = tmp_path / "track.npy"
    np.save(path, values)
    return path


def _base_argv(
    trebed: Path,
    track: Path,
    coordinate_opath: Path,
    mask_opath: Path,
    *,
    target_coord: int = 1,
    min_score: float = 0.0,
    strand: str = "+",
    relaxation: int = 0,
    track_window_size: int = 1,
    region_file_type: str = "TREbed",
) -> list[str]:
    return [
        "select_tss_relative_track",
        "--region_file_path",
        str(trebed),
        "--region_file_type",
        region_file_type,
        "--track_npy",
        str(track),
        "--strand",
        strand,
        "--target_coord",
        str(target_coord),
        "--relaxation",
        str(relaxation),
        # Use --flag=value so negative/infinite floats are not parsed as options.
        f"--min_score={min_score}",
        "--track_window_size",
        str(track_window_size),
        "--coordinate_opath",
        str(coordinate_opath),
        "--mask_opath",
        str(mask_opath),
    ]


def _default_track() -> np.ndarray:
    """Track for tiny.trebed: width 20 = max region length."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = 1.5  # row0 +1 at fwdTSS=105
    track[1, 5] = 9.0  # ignored when fwdTSS=-1
    track[2, 10] = 0.5  # row2 +1 at fwdTSS=310
    return track


def test_select_exact_plus_match_at_equality(tmp_path: Path):
    """Inclusive min_score match emits requested coord and true mask."""
    track_path = _build_track(tmp_path, _default_track())
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            coord_out,
            mask_out,
            target_coord=1,
            min_score=1.5,
        )
    )

    coords = np.load(coord_out)
    mask = np.load(mask_out)
    assert coords.shape == (3, 1)
    assert mask.shape == (3, 1)
    assert np.issubdtype(coords.dtype, np.integer)
    assert mask.dtype == np.bool_
    assert np.array_equal(coords.ravel(), [1, 0, 0])
    assert np.array_equal(mask.ravel(), [True, False, False])


def test_below_threshold_is_no_match(tmp_path: Path):
    """Score strictly below min_score yields coordinate 0 and mask false."""
    track = _default_track()
    track[0, 5] = 1.49
    track_path = _build_track(tmp_path, track)
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            coord_out,
            mask_out,
            target_coord=1,
            min_score=1.5,
        )
    )

    assert np.array_equal(np.load(coord_out).ravel(), [0, 0, 0])
    assert np.array_equal(np.load(mask_out).ravel(), [False, False, False])


def test_missing_fwd_tss_is_no_match_despite_track_score(tmp_path: Path):
    """fwdTSS=-1 yields no-match even when track cell would pass the cutoff."""
    track_path = _build_track(tmp_path, _default_track())
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            coord_out,
            mask_out,
            target_coord=1,
            min_score=0.0,
        )
    )

    coords = np.load(coord_out).ravel()
    mask = np.load(mask_out).ravel()
    assert coords[1] == 0
    assert not bool(mask[1])
    # Row2 passes with min_score 0.0; row0 also passes.
    assert coords[0] == 1 and bool(mask[0])
    assert coords[2] == 1 and bool(mask[2])


def test_outputs_roundtrip_through_public_annotation_apis(tmp_path: Path):
    """Emitted artifacts load as (N,1) stat/mask annotations in input order."""
    track_path = _build_track(tmp_path, _default_track())
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            coord_out,
            mask_out,
            target_coord=1,
            min_score=1.5,
        )
    )

    ge = GenomicElements(
        region_file_path=str(TINY_TREBED),
        region_file_type="TREbed",
        fasta_path=None,
    )
    ge.load_region_anno_from_npy("coord", str(coord_out), anno_type="stat")
    ge.load_region_anno_from_npy("mask", str(mask_out), anno_type="mask")
    coords = ge.get_stat_arr("coord")
    mask = ge.get_mask_arr("mask")
    assert coords.shape == (3, 1)
    assert mask.shape == (3, 1)
    assert np.array_equal(coords.ravel(), [1, 0, 0])
    assert np.array_equal(mask.ravel(), [True, False, False])
    assert np.array_equal(mask.ravel() == False, coords.ravel() == 0)


def test_rev_tss_irrelevant_for_plus_strand(tmp_path: Path):
    """Plus-strand selection ignores revTSS (row2 has both TSS values set)."""
    track = np.zeros((3, 20), dtype=float)
    # Place a high score at the revTSS-relative +1 index and a qualifying
    # score only at fwdTSS +1, confirming selection uses fwdTSS.
    track[2, 10] = 2.0  # fwdTSS=310 → index 10
    track[2, 15] = 99.0  # revTSS=315 → index 15 (must not be selected)
    track_path = _build_track(tmp_path, track)
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            coord_out,
            mask_out,
            target_coord=1,
            min_score=2.0,
        )
    )

    coords = np.load(coord_out).ravel()
    mask = np.load(mask_out).ravel()
    assert coords[2] == 1
    assert bool(mask[2])


def test_relaxed_window_selects_first_maximum(tmp_path: Path):
    """Ascending window scan keeps the first coordinate among equal maxima."""
    track = np.zeros((3, 20), dtype=float)
    # Window around target=-1 with r=1 is [-2, -1, 1].
    # Indices for row0 fwdTSS=105: -2→3, -1→4, +1→5
    track[0, 3] = 5.0
    track[0, 4] = 5.0  # equal max later → keep -2
    track[0, 5] = 1.0
    track_path = _build_track(tmp_path, track)
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            coord_out,
            mask_out,
            target_coord=-1,
            relaxation=1,
            min_score=5.0,
        )
    )

    coords = np.load(coord_out).ravel()
    mask = np.load(mask_out).ravel()
    assert coords[0] == -2
    assert bool(mask[0])
    assert coords[1] == 0 and not bool(mask[1])


def test_relaxed_window_crosses_tss_skipping_zero(tmp_path: Path):
    """Relaxed selection across the TSS uses no-zero window positions only."""
    track = np.zeros((3, 20), dtype=float)
    # target=1, r=1 → [-1, 1, 2]; indices 4, 5, 6 for fwdTSS=105
    track[0, 4] = 1.0
    track[0, 5] = 2.0
    track[0, 6] = 3.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            relaxation=1,
            min_score=2.5,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 2
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_inclusive_equality_on_selected_maximum(tmp_path: Path):
    """Match when the selected maximum equals min_score exactly."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 4] = 1.0
    track[0, 5] = 2.0
    track[0, 6] = 2.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            relaxation=1,
            min_score=2.0,
        )
    )

    # First max among equals at indices for [-1,1,2] is coord +1 (score 2)
    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_minus_strand_uses_rev_tss(tmp_path: Path):
    """Strand '-' selects revTSS; missing revTSS is no-match; fwdTSS ignored."""
    track = np.zeros((3, 20), dtype=float)
    # Row1 revTSS=205 → +1 index 5; row0 revTSS=-1; row2 revTSS=315 → index 15
    track[1, 5] = 4.0
    track[0, 5] = 99.0  # fwdTSS-relative cell must not matter
    track[2, 15] = 4.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            strand="-",
            target_coord=1,
            min_score=4.0,
        )
    )

    coords = np.load(tmp_path / "coord.npy").ravel()
    mask = np.load(tmp_path / "mask.npy").ravel()
    assert np.array_equal(coords, [0, 1, 1])
    assert np.array_equal(mask, [False, True, True])


def test_motif_minus_window_padding_semantics(tmp_path: Path):
    """Minus motif window indexes genomic-right 5-prime with W-1 padding."""
    # motif_window.trebed: [100,120) revTSS=110; W=3, coord=+1 → index 8
    track = np.zeros((1, 20), dtype=float)
    track[0, 8] = 7.5
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            MOTIF_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            strand="-",
            target_coord=1,
            min_score=7.5,
            track_window_size=3,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_motif_plus_window_uses_genomic_left_index(tmp_path: Path):
    """Plus motif window indexes the genomic-left 5-prime without padding shift."""
    track = np.zeros((1, 20), dtype=float)
    # fwdTSS=110, coord=+1, W=3 → index 10
    track[0, 10] = 3.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            MOTIF_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            strand="+",
            target_coord=1,
            min_score=3.0,
            track_window_size=3,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_row_order_preserved_with_mixed_matches(tmp_path: Path):
    """Output rows stay aligned to input TREbed order under mixed outcomes."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = 1.0
    track[2, 10] = 1.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            min_score=1.0,
        )
    )

    assert np.array_equal(np.load(tmp_path / "coord.npy").ravel(), [1, 0, 1])
    assert np.array_equal(np.load(tmp_path / "mask.npy").ravel(), [True, False, True])


def test_non_trebed_region_type_rejected(tmp_path: Path):
    track = np.zeros((3, 10), dtype=float)
    track_path = _build_track(tmp_path, track)
    with pytest.raises(ValueError, match="TREbed"):
        _run_cli(
            _base_argv(
                TINY_BED3,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                region_file_type="bed3",
            )
        )


def test_nonfinite_min_score_rejected(tmp_path: Path):
    track_path = _build_track(tmp_path, _default_track())
    with pytest.raises(ValueError, match="[Ff]inite"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                min_score=float("nan"),
            )
        )


def test_boolean_track_rejected(tmp_path: Path):
    track = np.zeros((3, 20), dtype=bool)
    track_path = _build_track(tmp_path, track)
    with pytest.raises(ValueError, match="[Bb]oolean|[Nn]umeric"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def _write_trebed(path: Path, rows: list[tuple]) -> Path:
    lines = ["\t".join(str(x) for x in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_both_tss_missing_is_row_level_no_match(tmp_path: Path):
    """Selected TSS -1 is no-match even when both TREbed TSS fields are missing."""
    trebed = _write_trebed(
        tmp_path / "both_missing.trebed",
        [("chr1", 100, 110, "b0", -1, -1)],
    )
    track = np.zeros((1, 10), dtype=float)
    track[0, 5] = 9.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            trebed,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            min_score=0.0,
        )
    )

    assert np.array_equal(np.load(tmp_path / "coord.npy").ravel(), [0])
    assert np.array_equal(np.load(tmp_path / "mask.npy").ravel(), [False])


def test_selected_tss_outside_interval_fails(tmp_path: Path):
    """Nonmissing selected TSS outside [start,end) fails with row/interval context."""
    trebed = _write_trebed(
        tmp_path / "oob_tss.trebed",
        [
            ("chr1", 100, 110, "ok", 105, -1),
            ("chr1", 200, 210, "bad", 210, -1),  # end is exclusive → outside
        ],
    )
    track = np.zeros((2, 10), dtype=float)
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"row 1|outside|\[200, 210\)"):
        _run_cli(
            _base_argv(
                trebed,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


@pytest.mark.parametrize(
    "tss,label",
    [
        (99, "before_start"),
        (110, "at_end_exclusive"),
    ],
)
def test_selected_tss_at_both_interval_edges_outside_fails(tmp_path: Path, tss, label):
    """TSS at start-1 and at end fail; start and end-1 remain valid elsewhere."""
    del label
    trebed = _write_trebed(
        tmp_path / "edge.trebed",
        [("chr1", 100, 110, "r", tss, -1)],
    )
    track = np.zeros((1, 10), dtype=float)
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"outside|row 0|\[100, 110\)"):
        _run_cli(
            _base_argv(
                trebed,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def test_unselected_invalid_tss_ignored_for_requested_strand(tmp_path: Path):
    """Crazy unselected revTSS must not affect plus-strand selection."""
    trebed = _write_trebed(
        tmp_path / "crazy_rev.trebed",
        [("chr1", 100, 110, "r", 105, 999999)],
    )
    track = np.zeros((1, 10), dtype=float)
    track[0, 5] = 2.0
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            trebed,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            strand="+",
            min_score=2.0,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_unavailable_relaxed_window_position_fails_with_row_coord_context(tmp_path: Path):
    """Any unavailable relaxed-window position fails (no clip/shrink/no-match)."""
    # Row length 10; fwdTSS=105; target=1,r=6 reaches far upstream/downstream.
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = 1.0
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"row 0|coord|unavailable|out of bounds|interval"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                target_coord=1,
                relaxation=6,
                min_score=0.0,
            )
        )


def test_short_region_unavailable_point_fails(tmp_path: Path):
    """Short intervals fail when the requested coordinate cannot fit."""
    trebed = _write_trebed(
        tmp_path / "short.trebed",
        [("chr1", 100, 103, "s", 101, -1)],
    )
    track = np.zeros((1, 3), dtype=float)
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"row 0|coord|unavailable|out of bounds|interval"):
        _run_cli(
            _base_argv(
                trebed,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                target_coord=5,
                min_score=0.0,
            )
        )


def test_storage_padding_beyond_logical_length_is_not_searched(tmp_path: Path):
    """Scores in max-width storage padding past the row length are never searched."""
    # tiny row0 length 10; track width 20. Padding index 15 must not win.
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = 1.0  # logical +1
    track[0, 15] = 100.0  # storage padding only
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            min_score=1.0,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_nan_in_padding_only_does_not_fail(tmp_path: Path):
    """NaN confined to unsearched storage padding is not an operation error."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = 1.0
    track[0, 15] = np.nan
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            min_score=1.0,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1


def test_searched_nan_fails_with_row_and_track_index(tmp_path: Path):
    """NaN in a searched cell fails with TREbed row and track-index context."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = np.nan
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"NaN|row 0|track index 5|index 5"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                target_coord=1,
                min_score=0.0,
            )
        )


def test_negative_infinity_score_is_unmatchable(tmp_path: Path):
    """-inf never satisfies a finite inclusive cutoff."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = float("-inf")
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            min_score=0.0,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 0
    assert not bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_positive_infinity_score_is_qualifying_maximum(tmp_path: Path):
    """+inf is a qualifying maximum against a finite cutoff."""
    track = np.zeros((3, 20), dtype=float)
    track[0, 4] = 10.0
    track[0, 5] = float("inf")
    track_path = _build_track(tmp_path, track)

    _run_cli(
        _base_argv(
            TINY_TREBED,
            track_path,
            tmp_path / "coord.npy",
            tmp_path / "mask.npy",
            target_coord=1,
            relaxation=1,
            min_score=0.0,
        )
    )

    assert np.load(tmp_path / "coord.npy").ravel()[0] == 1
    assert bool(np.load(tmp_path / "mask.npy").ravel()[0])


def test_nonnumeric_object_track_rejected(tmp_path: Path):
    # Unicode string arrays load without pickle but are nonnumeric.
    track = np.array([["x"] * 20] * 3)
    track_path = _build_track(tmp_path, track)
    with pytest.raises(ValueError, match="[Nn]umeric|[Bb]oolean"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def test_invalid_window_size_rejected(tmp_path: Path):
    track_path = _build_track(tmp_path, _default_track())
    with pytest.raises(ValueError, match="track_window_size"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                track_window_size=0,
            )
        )


def test_invalid_relaxation_rejected(tmp_path: Path):
    track_path = _build_track(tmp_path, _default_track())
    with pytest.raises(ValueError, match="relaxation"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                relaxation=-1,
            )
        )


def test_zero_target_coord_rejected(tmp_path: Path):
    track_path = _build_track(tmp_path, _default_track())
    with pytest.raises(ValueError, match="[Zz]ero"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                target_coord=0,
            )
        )


@pytest.mark.parametrize("bad_min", [float("inf"), float("-inf")])
def test_infinite_min_score_rejected(tmp_path: Path, bad_min: float):
    track_path = _build_track(tmp_path, _default_track())
    with pytest.raises(ValueError, match="[Ff]inite"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                min_score=bad_min,
            )
        )


def test_mismatched_track_row_count_rejected(tmp_path: Path):
    track = np.zeros((2, 20), dtype=float)
    track_path = _build_track(tmp_path, track)
    with pytest.raises(ValueError, match=r"regions|shape|match"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def test_mismatched_track_width_rejected(tmp_path: Path):
    track = np.zeros((3, 19), dtype=float)
    track_path = _build_track(tmp_path, track)
    with pytest.raises(ValueError, match=r"width|max region length|shape"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def test_malformed_1d_npy_track_rejected(tmp_path: Path):
    track_path = _build_track(tmp_path, np.zeros(20, dtype=float))
    with pytest.raises(ValueError, match=r"2D|shape"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def test_malformed_multi_array_npz_rejected(tmp_path: Path):
    npz_path = tmp_path / "multi.npz"
    np.savez(npz_path, a=np.zeros((3, 20)), b=np.zeros((3, 20)))
    with pytest.raises(ValueError, match=r"multiple arrays|NPZ"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                npz_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
            )
        )


def test_validation_failure_leaves_preexisting_outputs_unchanged(tmp_path: Path):
    """Scientific validation failures must not create or change final output paths."""
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"
    prior_coords = np.asarray([[7], [8], [9]], dtype=np.int64)
    prior_mask = np.asarray([[True], [False], [True]], dtype=bool)
    np.save(coord_out, prior_coords)
    np.save(mask_out, prior_mask)
    prior_coord_bytes = coord_out.read_bytes()
    prior_mask_bytes = mask_out.read_bytes()

    track = np.zeros((3, 20), dtype=float)
    track[0, 5] = np.nan
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"NaN|row 0"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                coord_out,
                mask_out,
                target_coord=1,
                min_score=0.0,
            )
        )

    assert coord_out.read_bytes() == prior_coord_bytes
    assert mask_out.read_bytes() == prior_mask_bytes


def test_unavailable_window_failure_does_not_create_outputs(tmp_path: Path):
    """Unavailable-window failures leave missing destinations uncreated."""
    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"
    track = np.zeros((3, 20), dtype=float)
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"row 0|coord|unavailable|out of bounds|interval"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                coord_out,
                mask_out,
                target_coord=1,
                relaxation=6,
                min_score=0.0,
            )
        )

    assert not coord_out.exists()
    assert not mask_out.exists()


def test_plus_and_minus_unavailable_windows_both_fail(tmp_path: Path):
    """Unavailable windows fail on both strands with actionable context."""
    track = np.zeros((3, 20), dtype=float)
    track_path = _build_track(tmp_path, track)

    with pytest.raises(ValueError, match=r"row|coord|unavailable|out of bounds|interval"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord_plus.npy",
                tmp_path / "mask_plus.npy",
                strand="+",
                target_coord=1,
                relaxation=6,
            )
        )

    with pytest.raises(ValueError, match=r"row|coord|unavailable|out of bounds|interval"):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord_minus.npy",
                tmp_path / "mask_minus.npy",
                strand="-",
                target_coord=1,
                relaxation=6,
            )
        )
