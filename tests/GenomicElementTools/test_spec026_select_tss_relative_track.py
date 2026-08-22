"""SPEC026 contract tests: GenomicElementTools select_tss_relative_track (ticket 01)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from GenomicElementTools.cli import GenomicElementTools
from RGTools.GenomicElements import GenomicElements

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tss_relative"
TINY_TREBED = FIXTURES / "tiny.trebed"
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
        "--min_score",
        str(min_score),
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


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"strand": "-"}, "[Nn]ot yet|[Uu]nsupported"),
        ({"relaxation": 1}, "[Nn]ot yet|[Uu]nsupported"),
        ({"track_window_size": 2}, "[Nn]ot yet|[Uu]nsupported"),
    ],
)
def test_later_modes_rejected_explicitly(tmp_path: Path, kwargs, match):
    track_path = _build_track(tmp_path, _default_track())
    with pytest.raises(ValueError, match=match):
        _run_cli(
            _base_argv(
                TINY_TREBED,
                track_path,
                tmp_path / "coord.npy",
                tmp_path / "mask.npy",
                **kwargs,
            )
        )


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
