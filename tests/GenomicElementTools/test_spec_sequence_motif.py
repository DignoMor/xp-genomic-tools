"""SPEC013 contract tests: GenomicElementTools onehot / motif_search / filter_motif_score."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from GenomicElementTools.cli import GenomicElementTools
from RGTools import GenomicElements, MemeMotif

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = FIXTURES / "spec"
GET = FIXTURES / "get"

TINY_FA = SPEC / "tiny.fa"
TINY_MEME = SPEC / "tiny.meme"
SAMPLE_MEME = FIXTURES / "sample-dna-motif.meme"
EQUAL_LEN_BED3 = GET / "equal_len.bed3"
REGIONS_BED3 = SPEC / "regions.bed3"


def _run(argv: list[str]):
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


# ---------------------------------------------------------------------------
# onehot
# ---------------------------------------------------------------------------


def test_onehot_shape_n_4_l_and_alphabet_order(tmp_path: Path):
    """SPEC013: get_all_region_one_hot then transpose → (N, 4, L); alphabet A,C,G,T."""
    out = tmp_path / "oh.npy"
    _run(
        [
            "onehot",
            "--fasta_path",
            str(TINY_FA),
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--opath",
            str(out),
        ]
    )
    arr = np.load(out)
    assert arr.shape == (2, 4, 4)

    # equal_len.bed3: chrA:0-4 = ACGT; chrA:4-8 = ACGT
    expected_acgt = np.eye(4, dtype=arr.dtype)  # rows A,C,G,T as one-hot columns per position
    # Stored as (4, L): channel axis first after transpose(0,2,1) from (L,4)
    for i in range(2):
        np.testing.assert_array_equal(arr[i], expected_acgt)

    ge = GenomicElements(str(EQUAL_LEN_BED3), "bed3", str(TINY_FA))
    try:
        raw = ge.get_all_region_one_hot()
        np.testing.assert_array_equal(arr, raw.transpose(0, 2, 1))
    finally:
        ge.close()


def test_onehot_preserves_region_order(tmp_path: Path):
    """regions.bed3 order is chrB then chrA (not chrom-sorted)."""
    out = tmp_path / "oh.npy"
    _run(
        [
            "onehot",
            "--fasta_path",
            str(TINY_FA),
            "--region_file_path",
            str(REGIONS_BED3),
            "--region_file_type",
            "bed3",
            "--opath",
            str(out),
        ]
    )
    arr = np.load(out)
    assert arr.shape == (2, 4, 4)
    # chrB:1-5 = GGGC
    gggc = np.array(
        [
            [0, 0, 0, 0],  # A
            [0, 0, 0, 1],  # C
            [1, 1, 1, 0],  # G
            [0, 0, 0, 0],  # T
        ],
        dtype=arr.dtype,
    )
    # chrA:0-4 = ACGT
    acgt = np.eye(4, dtype=arr.dtype)
    np.testing.assert_array_equal(arr[0], gggc)
    np.testing.assert_array_equal(arr[1], acgt)


def test_onehot_rejects_mixed_region_lengths(tmp_path: Path):
    """SPEC013 Planned gap: equal-length intended; library raises on mixed lengths."""
    bed = tmp_path / "mixed.bed3"
    bed.write_text("chrA\t0\t4\nchrA\t0\t8\n")
    with pytest.raises(ValueError, match="length-homogeneous"):
        _run(
            [
                "onehot",
                "--fasta_path",
                str(TINY_FA),
                "--region_file_path",
                str(bed),
                "--region_file_type",
                "bed3",
                "--opath",
                str(tmp_path / "oh.npy"),
            ]
        )


# ---------------------------------------------------------------------------
# motif_search
# ---------------------------------------------------------------------------


def test_motif_search_writes_one_npy_per_motif(tmp_path: Path):
    header = tmp_path / "ms"
    _run(
        [
            "motif_search",
            "--fasta_path",
            str(TINY_FA),
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--motif_file",
            str(TINY_MEME),
            "--output_header",
            str(header),
            "--estimate_background_freq",
            "False",
            "--strand",
            "+",
        ]
    )
    mm = MemeMotif(str(TINY_MEME))
    for name in mm.get_motif_list():
        path = Path(f"{header}.{name}.npy")
        assert path.is_file(), f"missing {path}"
        track = np.load(path)
        assert track.shape[0] == 2  # one track vector per region
        assert track.ndim == 2


def test_motif_search_track_aligned_to_regions_and_search_api(tmp_path: Path):
    """One track row per region; scores agree with MemeMotif.search_one_motif after SPEC013 pseudocount."""
    header = tmp_path / "ms"
    _run(
        [
            "motif_search",
            "--fasta_path",
            str(TINY_FA),
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--motif_file",
            str(TINY_MEME),
            "--output_header",
            str(header),
            "--estimate_background_freq",
            "False",
            "--strand",
            "+",
        ]
    )
    mm = MemeMotif(str(TINY_MEME))
    name = mm.get_motif_list()[0]
    alphabet = mm.get_alphabet()
    bg = np.asarray(mm.get_bg_freq(), dtype=float)
    raw_pwm = mm.get_motif_pwm(name)
    # SPEC013: (counts + 1) / (total_sites + alphabet_size). File PWMs are already
    # probabilities; recover counts via nsites then re-normalize with pseudocount.
    nsites = mm.get_motif_num_source_sites(name)
    alen = mm.get_motif_alphabet_length(name)
    counts = raw_pwm * nsites
    pwm = (counts + 1.0) / (nsites + alen)

    ge = GenomicElements(str(EQUAL_LEN_BED3), "bed3", str(TINY_FA))
    try:
        seqs = ge.get_all_region_seqs()
        written = np.load(f"{header}.{name}.npy")
        assert written.shape[0] == len(seqs)
        for i, seq in enumerate(seqs):
            lib = MemeMotif.search_one_motif(
                seq, alphabet, pwm, bg_freq=bg.tolist(), strand="+"
            )
            np.testing.assert_allclose(written[i], lib, rtol=1e-5, atol=1e-5)
    finally:
        ge.close()


def test_motif_search_strand_both_writes_output(tmp_path: Path):
    header = tmp_path / "motif_search"
    _run(
        [
            "motif_search",
            "--fasta_path",
            str(TINY_FA),
            "--region_file_path",
            str(EQUAL_LEN_BED3),
            "--region_file_type",
            "bed3",
            "--motif_file",
            str(TINY_MEME),
            "--output_header",
            str(header),
            "--strand",
            "both",
        ]
    )
    assert Path(f"{header}.SPEC_TINY.npy").is_file()


def test_sample_dna_motif_fixture_too_wide_for_tiny_beds():
    """sample-dna-motif.meme is available but motifs exceed equal_len.bed3 width."""
    assert SAMPLE_MEME.is_file()
    mm = MemeMotif(str(SAMPLE_MEME))
    max_w = max(mm.get_motif_length(n) for n in mm.get_motif_list())
    assert max_w > 4


# ---------------------------------------------------------------------------
# filter_motif_score — strict inequalities
# ---------------------------------------------------------------------------


def test_filter_motif_score_strict_inequalities(tmp_path: Path):
    """Keep regions where min_score < track[filter_base] < max_score (strict)."""
    bed = tmp_path / "regions.bed3"
    bed.write_text("chrA\t0\t4\nchrA\t4\t8\n")
    # filter_base=1 → scores 1.0 and 5.0
    track = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 5.0, 2.0, 3.0]], dtype=float)
    npy = tmp_path / "motif.npy"
    np.save(npy, track)

    header = tmp_path / "filt"
    _run(
        [
            "filter_motif_score",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed3",
            "--motif_search_npy",
            str(npy),
            "--output_header",
            str(header),
            "--filter_base",
            "1",
            "--min_score",
            "0.9",
            "--max_score",
            "5.0",
        ]
    )
    # 1.0 is inside (0.9, 5.0); 5.0 is NOT < 5.0 → only first region
    out_bed = Path(f"{header}.bed")
    out_motif = Path(f"{header}.motif.npy")
    assert out_bed.is_file()
    assert out_motif.is_file()
    lines = [ln for ln in out_bed.read_text().splitlines() if ln.strip()]
    assert lines == ["chrA\t0\t4"]
    filtered = np.load(out_motif)
    assert filtered.shape[0] == 1
    np.testing.assert_allclose(filtered[0], [0.0, 1.0, 2.0, 3.0])


@pytest.mark.xfail(
    reason=(
        "SPEC013 allows empty filter result; GenomicElements.apply_logical_filter "
        "currently raises on empty region table (max length NAType)"
    ),
    raises=(TypeError, ValueError),
    strict=True,
)
def test_filter_motif_score_empty_result_writes_outputs(tmp_path: Path):
    bed = tmp_path / "regions.bed3"
    bed.write_text("chrA\t0\t4\nchrA\t4\t8\n")
    track = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], dtype=float)
    npy = tmp_path / "motif.npy"
    np.save(npy, track)
    header = tmp_path / "filt_empty"
    _run(
        [
            "filter_motif_score",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed3",
            "--motif_search_npy",
            str(npy),
            "--output_header",
            str(header),
            "--filter_base",
            "1",
            "--min_score",
            "1.0",
            "--max_score",
            "5.0",
        ]
    )
    assert Path(f"{header}.bed").is_file()
    assert Path(f"{header}.motif.npy").is_file()
    assert Path(f"{header}.bed").read_text().strip() == ""
    assert np.load(f"{header}.motif.npy").shape[0] == 0


def test_filter_motif_score_filter_base_out_of_range(tmp_path: Path):
    bed = tmp_path / "regions.bed3"
    bed.write_text("chrA\t0\t4\n")
    track = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=float)
    npy = tmp_path / "motif.npy"
    np.save(npy, track)
    with pytest.raises(IndexError):
        _run(
            [
                "filter_motif_score",
                "--region_file_path",
                str(bed),
                "--region_file_type",
                "bed3",
                "--motif_search_npy",
                str(npy),
                "--output_header",
                str(tmp_path / "bad"),
                "--filter_base",
                "99",
                # Use '=' form so argparse does not treat "-inf" as a flag.
                "--min_score=-inf",
                "--max_score=inf",
            ]
        )
