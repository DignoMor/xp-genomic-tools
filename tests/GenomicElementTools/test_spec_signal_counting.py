"""SPEC012 contract tests: GenomicElementTools count_single_bw / count_paired_bw."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

pyBigWig = pytest.importorskip("pyBigWig")

from GenomicElementTools.cli import GenomicElementTools
from RGTools import GenomicElements, PairedBwTrack, SingleBwTrack

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = FIXTURES / "spec"
GET = FIXTURES / "get"

EQUAL_LEN_BED3 = GET / "equal_len.bed3"
STRANDED_BED6 = GET / "stranded.bed6"
REGIONS_BED3 = SPEC / "regions.bed3"


def _run(argv: list[str]):
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _write_constant_bw(path: Path, chrom: str, chrom_size: int, value: float) -> Path:
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, chrom_size)])
    bw.addEntries([chrom], [0], ends=[chrom_size], values=[float(value)])
    bw.close()
    return path


def _write_multi_chrom_bw(path: Path, chrom_values: dict[str, float], chrom_size: int = 8) -> Path:
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(c, chrom_size) for c in chrom_values])
    chroms = list(chrom_values)
    bw.addEntries(
        chroms,
        [0] * len(chroms),
        ends=[chrom_size] * len(chroms),
        values=[float(chrom_values[c]) for c in chroms],
    )
    bw.close()
    return path


# ---------------------------------------------------------------------------
# count_single_bw
# ---------------------------------------------------------------------------


def test_count_single_bw_raw_count_order_and_stat_shape(tmp_path: Path):
    """One scalar per region in input order; non-full_track → stat-shaped (N, 1)."""
    bw = _write_multi_chrom_bw(tmp_path / "sig.bw", {"chrA": 1.0, "chrB": 2.0})
    out = tmp_path / "count.npy"
    _run(
        [
            "count_single_bw",
            "--region_file_path",
            str(REGIONS_BED3),
            "--region_file_type",
            "bed3",
            "--bw_path",
            str(bw),
            "--quantification_type",
            "raw_count",
            "--opath",
            str(out),
        ]
    )
    arr = np.load(out)
    assert arr.shape == (2, 1)
    # regions.bed3: chrB:1-5 (len 4) @ 2.0 → 8; chrA:0-4 (len 4) @ 1.0 → 4
    np.testing.assert_allclose(arr.ravel(), [8.0, 4.0])

    ge = GenomicElements(str(REGIONS_BED3), "bed3", fasta_path=None)
    try:
        ge.load_region_anno_from_npy("count", str(out), anno_type="stat")
        assert ge.get_anno_type("count") == "stat"
        np.testing.assert_allclose(ge.get_stat_arr("count").ravel(), [8.0, 4.0])
    finally:
        ge.close()


def test_count_single_bw_full_track_saves_track_shape(tmp_path: Path):
    bw = _write_constant_bw(tmp_path / "sig.bw", "chrA", 8, 1.5)
    out = tmp_path / "count.npz"
    _run(
        [
            "count_single_bw",
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--bw_path",
            str(bw),
            "--quantification_type",
            "full_track",
            "--opath",
            str(out),
        ]
    )
    with np.load(out) as z:
        assert list(z.keys()) == ["arr_0"]
        arr = z["arr_0"]
    assert arr.shape == (2, 4)
    np.testing.assert_allclose(arr, 1.5)

    ge = GenomicElements(str(EQUAL_LEN_BED3), "bed3", fasta_path=None)
    try:
        ge.load_region_anno_from_npy("count", str(out), anno_type="track")
        assert ge.get_anno_type("count") == "track"
    finally:
        ge.close()


def test_count_single_bw_opath_npz_vs_other_suffix(tmp_path: Path):
    """Only ``.npz`` selects npz save; other suffixes use npy (may append ``.npy``)."""
    bw = _write_constant_bw(tmp_path / "sig.bw", "chrA", 8, 1.0)
    npz_path = tmp_path / "out.npz"
    other = tmp_path / "out.counts"
    for opath in (npz_path, other):
        _run(
            [
                "count_single_bw",
                "--region_file_path",
                str(EQUAL_LEN_BED3),
                "--region_file_type",
                "bed3",
                "--bw_path",
                str(bw),
                "--opath",
                str(opath),
            ]
        )
    assert npz_path.is_file()
    with np.load(npz_path) as z:
        assert "arr_0" in z
    # Non-.npz paths go through npy save; implementation appends .npy when needed.
    npy_sibling = Path(str(other) + ".npy")
    assert other.is_file() or npy_sibling.is_file()


def test_count_single_bw_matches_SingleBwTrack(tmp_path: Path):
    bw = _write_constant_bw(tmp_path / "sig.bw", "chrA", 8, 3.0)
    out = tmp_path / "count.npy"
    _run(
        [
            "count_single_bw",
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--bw_path",
            str(bw),
            "--quantification_type",
            "RPK",
            "--opath",
            str(out),
        ]
    )
    track = SingleBwTrack(str(bw))
    expected = [
        track.count_single_region("chrA", 0, 4, output_type="RPK", min_len_after_padding=1),
        track.count_single_region("chrA", 4, 8, output_type="RPK", min_len_after_padding=1),
    ]
    np.testing.assert_allclose(np.load(out).ravel(), expected)


def test_count_single_bw_invalid_quantification_type(tmp_path: Path):
    bw = _write_constant_bw(tmp_path / "sig.bw", "chrA", 8, 1.0)
    with pytest.raises(SystemExit):
        _run(
            [
                "count_single_bw",
                "--region_file_path",
                str(EQUAL_LEN_BED3),
                "--region_file_type",
                "bed3",
                "--bw_path",
                str(bw),
                "--quantification_type",
                "not_a_type",
                "--opath",
                str(tmp_path / "out.npy"),
            ]
        )


# ---------------------------------------------------------------------------
# count_paired_bw — strand resolution (SPEC012)
# ---------------------------------------------------------------------------


def test_count_paired_bw_strand_from_bed6(tmp_path: Path):
    pl = _write_constant_bw(tmp_path / "pl.bw", "chrA", 8, 10.0)
    mn = _write_constant_bw(tmp_path / "mn.bw", "chrA", 8, 2.0)
    out = tmp_path / "paired.npy"
    _run(
        [
            "count_paired_bw",
            "--region_file_path",
            str(STRANDED_BED6),
            "--region_file_type",
            "bed6",
            "--bw_pl",
            str(pl),
            "--bw_mn",
            str(mn),
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--opath",
            str(out),
        ]
    )
    # len-4 windows: + → 40; − → −8; . → 32
    np.testing.assert_allclose(np.load(out).ravel(), [40.0, -8.0, 32.0])


def test_count_paired_bw_override_strand(tmp_path: Path):
    pl = _write_constant_bw(tmp_path / "pl.bw", "chrA", 8, 10.0)
    mn = _write_constant_bw(tmp_path / "mn.bw", "chrA", 8, 2.0)
    out = tmp_path / "paired.npy"
    _run(
        [
            "count_paired_bw",
            "--region_file_path",
            str(STRANDED_BED6),
            "--region_file_type",
            "bed6",
            "--bw_pl",
            str(pl),
            "--bw_mn",
            str(mn),
            "--override_strand",
            "+",
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--opath",
            str(out),
        ]
    )
    np.testing.assert_allclose(np.load(out).ravel(), [40.0, 40.0, 40.0])


def test_count_paired_bw_bed3_uses_unstranded_dot(tmp_path: Path):
    pl = _write_constant_bw(tmp_path / "pl.bw", "chrA", 8, 10.0)
    mn = _write_constant_bw(tmp_path / "mn.bw", "chrA", 8, 2.0)
    out = tmp_path / "paired.npy"
    _run(
        [
            "count_paired_bw",
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--bw_pl",
            str(pl),
            "--bw_mn",
            str(mn),
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--opath",
            str(out),
        ]
    )
    # bed3 → strand "."; both regions len 4 → 32 each
    np.testing.assert_allclose(np.load(out).ravel(), [32.0, 32.0])


def test_count_paired_bw_matches_PairedBwTrack(tmp_path: Path):
    pl = _write_constant_bw(tmp_path / "pl.bw", "chrA", 8, 10.0)
    mn = _write_constant_bw(tmp_path / "mn.bw", "chrA", 8, 2.0)
    out = tmp_path / "paired.npy"
    _run(
        [
            "count_paired_bw",
            "--region_file_path",
            str(STRANDED_BED6),
            "--region_file_type",
            "bed6",
            "--bw_pl",
            str(pl),
            "--bw_mn",
            str(mn),
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--quantification_type",
            "raw_count",
            "--opath",
            str(out),
        ]
    )
    track = PairedBwTrack(str(pl), str(mn))
    expected = [
        track.count_single_region(
            "chrA", 0, 4, strand="+", output_type="raw_count", min_len_after_padding=1
        ),
        track.count_single_region(
            "chrA", 0, 4, strand="-", output_type="raw_count", min_len_after_padding=1
        ),
        track.count_single_region(
            "chrA", 0, 4, strand=".", output_type="raw_count", min_len_after_padding=1
        ),
    ]
    np.testing.assert_allclose(np.load(out).ravel(), expected)


def test_count_paired_bw_full_track_npz(tmp_path: Path):
    pl = _write_constant_bw(tmp_path / "pl.bw", "chrA", 8, 1.0)
    mn = _write_constant_bw(tmp_path / "mn.bw", "chrA", 8, 0.0)
    out = tmp_path / "paired.npz"
    _run(
        [
            "count_paired_bw",
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--bw_pl",
            str(pl),
            "--bw_mn",
            str(mn),
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--quantification_type",
            "full_track",
            "--opath",
            str(out),
        ]
    )
    with np.load(out) as z:
        arr = z["arr_0"]
    assert arr.shape[0] == 2
    assert arr.ndim == 2
