"""SPEC002 / SPEC007 black-box tests for SingleBwTrack / PairedBwTrack."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pyBigWig = pytest.importorskip("pyBigWig")

from RGTools import PairedBwTrack, SingleBwTrack
from RGTools.BwTrack import BaseBwTrack


def _write_constant_bw(path: Path, chrom: str, chrom_size: int, start: int, end: int, value: float):
    """Write a minimal BigWig with one constant interval [start, end)."""
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, chrom_size)])
    bw.addEntries([chrom], [start], ends=[end], values=[float(value)])
    bw.close()


@pytest.fixture
def tiny_bw(tmp_path: Path) -> Path:
    """chr1:1000bp; signal 1.0 on [10,20), 2.0 on [20,30), 3.0 on [30,40)."""
    path = tmp_path / "tiny.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries(
        ["chr1", "chr1", "chr1"],
        [10, 20, 30],
        ends=[20, 30, 40],
        values=[1.0, 2.0, 3.0],
    )
    bw.close()
    return path


@pytest.fixture
def paired_bws(tmp_path: Path) -> tuple[Path, Path]:
    pl = tmp_path / "plus.bw"
    mn = tmp_path / "minus.bw"
    _write_constant_bw(pl, "chr1", 200, 20, 50, 10.0)
    _write_constant_bw(mn, "chr1", 200, 20, 50, 2.0)
    return pl, mn


def test_supported_quantification_types():
    assert BaseBwTrack.get_supported_quantification_type() == [
        "raw_count",
        "RPK",
        "full_track",
    ]


def test_quantification_types_raw_rpk_full_track(tiny_bw: Path):
    track = SingleBwTrack(str(tiny_bw))
    # Interval [10, 40): ten 1s + ten 2s + ten 3s
    raw = track.count_single_region(
        "chr1", 10, 40, output_type="raw_count", min_len_after_padding=1
    )
    assert raw == pytest.approx(10 * 1.0 + 10 * 2.0 + 10 * 3.0)

    rpk = track.count_single_region("chr1", 10, 40, output_type="RPK", min_len_after_padding=1)
    assert rpk == pytest.approx(raw / 30.0 * 1e3)

    full = track.count_single_region(
        "chr1", 10, 40, output_type="full_track", min_len_after_padding=1
    )
    assert isinstance(full, np.ndarray)
    assert full.shape == (30,)
    np.testing.assert_allclose(full[:10], 1.0)
    np.testing.assert_allclose(full[10:20], 2.0)
    np.testing.assert_allclose(full[20:30], 3.0)


def test_missing_chrom_returns_zeros(tiny_bw: Path):
    track = SingleBwTrack(str(tiny_bw))
    raw = track.count_single_region(
        "chrMissing", 0, 10, output_type="raw_count", min_len_after_padding=1
    )
    assert raw == pytest.approx(0.0)

    full = track.count_single_region(
        "chrMissing", 0, 10, output_type="full_track", min_len_after_padding=1
    )
    assert isinstance(full, np.ndarray)
    assert full.shape == (10,)
    np.testing.assert_allclose(full, 0.0)


def test_padding_fallback_raise_drop(tiny_bw: Path):
    track = SingleBwTrack(str(tiny_bw))
    # Region length 10; pads +10 → padded length 20 < min_len_after_padding=50 → invalid.
    region = dict(chrom="chr1", start=10, end=20, output_type="raw_count", l_pad=5, r_pad=5)
    baseline = track.count_single_region(
        "chr1", 10, 20, output_type="raw_count", min_len_after_padding=1
    )

    fallback = track.count_single_region(
        **region, min_len_after_padding=50, method_resolving_invalid_padding="fallback"
    )
    assert fallback == pytest.approx(baseline)

    with pytest.raises(Exception):
        track.count_single_region(
            **region, min_len_after_padding=50, method_resolving_invalid_padding="raise"
        )

    dropped = track.count_single_region(
        **region, min_len_after_padding=50, method_resolving_invalid_padding="drop"
    )
    assert isinstance(dropped, float) and math.isnan(dropped)


def test_padding_applies_when_valid(tiny_bw: Path):
    track = SingleBwTrack(str(tiny_bw))
    # [10,20) with ±5 pad → [5,25): five 0s + ten 1s + five 2s = 20
    padded = track.count_single_region(
        "chr1",
        10,
        20,
        output_type="raw_count",
        l_pad=5,
        r_pad=5,
        min_len_after_padding=20,
        method_resolving_invalid_padding="raise",
    )
    assert padded == pytest.approx(20.0)


def test_paired_bw_strand_plus_minus_unstranded(paired_bws: tuple[Path, Path]):
    pl, mn = paired_bws
    track = PairedBwTrack(str(pl), str(mn))
    kwargs = dict(
        chrom="chr1",
        start=20,
        end=50,
        output_type="raw_count",
        min_len_after_padding=1,
    )
    # 30 bases × 10 on plus; 30 × 2 on minus (negated by default).
    plus = track.count_single_region(strand="+", **kwargs)
    minus = track.count_single_region(strand="-", **kwargs)
    both = track.count_single_region(strand=".", **kwargs)

    assert plus == pytest.approx(300.0)
    assert minus == pytest.approx(-60.0)
    assert both == pytest.approx(240.0)


def test_paired_bw_invalid_strand(paired_bws: tuple[Path, Path]):
    pl, mn = paired_bws
    track = PairedBwTrack(str(pl), str(mn))
    with pytest.raises(Exception):
        track.count_single_region(
            "chr1", 20, 50, strand="x", output_type="raw_count", min_len_after_padding=1
        )


def test_unsupported_quantification_type(tiny_bw: Path):
    track = SingleBwTrack(str(tiny_bw))
    with pytest.raises(Exception):
        track.count_single_region(
            "chr1", 10, 20, output_type="not_a_type", min_len_after_padding=1
        )
