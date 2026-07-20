"""SPEC011 contract tests: GenomicElementTools region transforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from GenomicElementTools import GenomicElementTools
from RGTools.exceptions import InvalidBedRegionException

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "get"
TINY_BED6 = FIXTURES / "tiny.bed6"


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _read_bed_lines(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# pad_region
# ---------------------------------------------------------------------------


def test_pad_region_preserves_order(tmp_path: Path):
    """Padded output keeps input row order (SPEC011)."""
    bed = tmp_path / "order.bed3"
    bed.write_text("chrB\t10\t20\nchrA\t50\t60\n")
    out = tmp_path / "padded.bed3"

    _run_cli(
        [
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed3",
            "--upstream_pad",
            "2",
            "--downstream_pad",
            "3",
            "--ignore_strand",
            "true",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert [r[0] for r in rows] == ["chrB", "chrA"]
    assert rows[0][1:3] == ["8", "23"]
    assert rows[1][1:3] == ["48", "63"]


def test_pad_region_fallback_keeps_original_on_invalid(tmp_path: Path):
    """fallback keeps the original region when padding is invalid (SPEC011)."""
    bed = tmp_path / "in.bed6"
    # First pads cleanly; second goes negative on the start coordinate.
    bed.write_text("chr1\t10\t20\tA\t1\t+\nchr1\t0\t5\tB\t1\t-\n")
    out = tmp_path / "fallback.bed6"

    _run_cli(
        [
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed6",
            "--upstream_pad",
            "5",
            "--downstream_pad",
            "10",
            "--method_resolving_invalid_region",
            "fallback",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert len(rows) == 2
    assert rows[0][1:3] == ["5", "30"]
    assert rows[1][1:3] == ["0", "5"]
    assert rows[1][3] == "B"


def test_pad_region_drop_omits_invalid(tmp_path: Path):
    """drop omits regions that become invalid after padding (SPEC011)."""
    bed = tmp_path / "in.bed6"
    bed.write_text("chr1\t10\t20\tA\t1\t+\nchr1\t0\t5\tB\t1\t-\n")
    out = tmp_path / "drop.bed6"

    _run_cli(
        [
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed6",
            "--upstream_pad",
            "5",
            "--downstream_pad",
            "10",
            "--method_resolving_invalid_region",
            "drop",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert len(rows) == 1
    assert rows[0][3] == "A"
    assert rows[0][1:3] == ["5", "30"]


def test_pad_region_raise_rethrows_invalid(tmp_path: Path):
    """raise rethrows InvalidBedRegionException (SPEC011)."""
    bed = tmp_path / "in.bed6"
    bed.write_text("chr1\t0\t5\tB\t1\t-\n")
    out = tmp_path / "raise.bed6"

    with pytest.raises(InvalidBedRegionException):
        _run_cli(
            [
                "pad_region",
                "--region_file_path",
                str(bed),
                "--region_file_type",
                "bed6",
                "--upstream_pad",
                "5",
                "--downstream_pad",
                "10",
                "--method_resolving_invalid_region",
                "raise",
                "--opath",
                str(out),
            ]
        )


# ---------------------------------------------------------------------------
# bed2tssbed
# ---------------------------------------------------------------------------


def test_bed2tssbed_tss_plus_and_minus(tmp_path: Path):
    """TSS site is start on + and end-1 on - (SPEC011)."""
    out = tmp_path / "tss.bed6"
    _run_cli(
        [
            "bed2tssbed",
            "--region_file_path",
            str(TINY_BED6),
            "--region_file_type",
            "bed6",
            "--output_site",
            "TSS",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert len(rows) == 2
    # + : [10, 11); - : [39, 40)
    assert rows[0][1:4] == ["10", "11", "peakA"]
    assert rows[0][5] == "+"
    assert rows[1][1:4] == ["39", "40", "peakB"]
    assert rows[1][5] == "-"


def test_bed2tssbed_center(tmp_path: Path):
    """center site is (start + end) // 2 as a length-1 interval (SPEC011)."""
    out = tmp_path / "center.bed6"
    _run_cli(
        [
            "bed2tssbed",
            "--region_file_path",
            str(TINY_BED6),
            "--region_file_type",
            "bed6",
            "--output_site",
            "center",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert rows[0][1:3] == ["15", "16"]
    assert rows[1][1:3] == ["35", "36"]


# ---------------------------------------------------------------------------
# track2tss_bed
# ---------------------------------------------------------------------------


def test_track2tss_bed_max_abs_sig(tmp_path: Path):
    """MaxAbsSig site = region.start + argmax(|track|) (SPEC011)."""
    # Region lengths are both 10; peaks at offsets 3 and 7 (negative).
    track = np.zeros((2, 10), dtype=float)
    track[0, 3] = 5.0
    track[1, 7] = -9.0
    track_path = tmp_path / "track.npy"
    np.save(track_path, track)
    out = tmp_path / "track_tss.bed6"

    _run_cli(
        [
            "track2tss_bed",
            "--region_file_path",
            str(TINY_BED6),
            "--region_file_type",
            "bed6",
            "--track",
            str(track_path),
            "--output_site",
            "MaxAbsSig",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert rows[0][1:3] == ["13", "14"]
    assert rows[1][1:3] == ["37", "38"]


# ---------------------------------------------------------------------------
# get_context_ge
# ---------------------------------------------------------------------------


def test_get_context_ge_nearest(tmp_path: Path):
    """nearest picks same-chrom context minimizing closest-end distance (SPEC011)."""
    query = tmp_path / "query.bed3"
    query.write_text("chr1\t100\t110\nchr1\t200\t210\n")
    context = tmp_path / "context.bed3"
    context.write_text("chr1\t50\t60\nchr1\t180\t190\nchr1\t220\t230\n")
    out = tmp_path / "nearest.bed3"

    _run_cli(
        [
            "get_context_ge",
            "nearest",
            "--region_file_path",
            str(query),
            "--region_file_type",
            "bed3",
            "--context_file_path",
            str(context),
            "--context_file_type",
            "bed3",
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert rows == [["chr1", "50", "60"], ["chr1", "180", "190"]]


def test_get_context_ge_nearest_empty_context_raises(tmp_path: Path):
    """No context on the query chromosome → ValueError (SPEC011)."""
    query = tmp_path / "query.bed3"
    query.write_text("chr2\t10\t20\n")
    context = tmp_path / "context.bed3"
    context.write_text("chr1\t50\t60\n")
    out = tmp_path / "nearest.bed3"

    with pytest.raises(ValueError, match="[Nn]o context"):
        _run_cli(
            [
                "get_context_ge",
                "nearest",
                "--region_file_path",
                str(query),
                "--region_file_type",
                "bed3",
                "--context_file_path",
                str(context),
                "--context_file_type",
                "bed3",
                "--opath",
                str(out),
            ]
        )


def test_get_context_ge_windowed_argmax(tmp_path: Path):
    """windowed_argmax selects contained context with max stat (SPEC011)."""
    query = tmp_path / "windows.bed3"
    query.write_text("chr1\t40\t200\nchr1\t210\t250\n")
    context = tmp_path / "context.bed3"
    context.write_text("chr1\t50\t60\nchr1\t180\t190\nchr1\t220\t230\n")
    stat_path = tmp_path / "context_stat.npy"
    np.save(stat_path, np.array([1.0, 5.0, 3.0]))
    out = tmp_path / "wargmax.bed3"

    _run_cli(
        [
            "get_context_ge",
            "windowed_argmax",
            "--region_file_path",
            str(query),
            "--region_file_type",
            "bed3",
            "--context_file_path",
            str(context),
            "--context_file_type",
            "bed3",
            "--context_stat_path",
            str(stat_path),
            "--opath",
            str(out),
        ]
    )

    rows = _read_bed_lines(out)
    assert rows == [["chr1", "180", "190"], ["chr1", "220", "230"]]


def test_get_context_ge_windowed_argmax_empty_window_raises(tmp_path: Path):
    """Empty window (no fully contained context) → ValueError (SPEC011)."""
    query = tmp_path / "windows.bed3"
    query.write_text("chr1\t0\t10\n")
    context = tmp_path / "context.bed3"
    context.write_text("chr1\t50\t60\n")
    stat_path = tmp_path / "context_stat.npy"
    np.save(stat_path, np.array([1.0]))
    out = tmp_path / "wargmax.bed3"

    with pytest.raises(ValueError, match="[Nn]o context"):
        _run_cli(
            [
                "get_context_ge",
                "windowed_argmax",
                "--region_file_path",
                str(query),
                "--region_file_type",
                "bed3",
                "--context_file_path",
                str(context),
                "--context_file_type",
                "bed3",
                "--context_stat_path",
                str(stat_path),
                "--opath",
                str(out),
            ]
        )
