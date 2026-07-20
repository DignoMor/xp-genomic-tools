"""SPEC020 contract tests: ExogeneousSequenceTools onehot / motif_search."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from ExogeneousSequenceTools.Motif import Motif
from ExogeneousSequenceTools.cli import ExogeneousSequenceTools
from RGTools.ExogeneousSequences import ExogeneousSequences
from RGTools.MemeMotif import MemeMotif

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPEC = FIXTURES / "spec"

TINY_FA = SPEC / "tiny.fa"
TINY_MEME = SPEC / "tiny.meme"
SAMPLE_MEME = FIXTURES / "sample-dna-motif.meme"

_SKIP_NO_MEME = "Requires tests/fixtures/spec/tiny.meme or sample-dna-motif.meme"


def _meme_fixture() -> Path:
    if TINY_MEME.is_file():
        return TINY_MEME
    if SAMPLE_MEME.is_file():
        return SAMPLE_MEME
    return TINY_MEME


def _run(argv: list[str]):
    parser = argparse.ArgumentParser()
    ExogeneousSequenceTools.set_parser(parser)
    args = parser.parse_args(argv)
    ExogeneousSequenceTools.main(args)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    ExogeneousSequences.write_sequences_to_fasta(
        [r[0] for r in records],
        [r[1] for r in records],
        str(path),
    )


# ---------------------------------------------------------------------------
# onehot
# ---------------------------------------------------------------------------


def test_onehot_shape_n_4_l_channel_first(tmp_path: Path):
    """SPEC020: get_all_region_one_hot then transpose → (N, 4, L)."""
    fasta = tmp_path / "equal.fa"
    _write_fasta(
        fasta,
        [
            ("seq1", "ACGT"),
            ("seq2", "TGCA"),
        ],
    )
    out = tmp_path / "oh.npy"
    _run(["onehot", "--fasta", str(fasta), "--opath", str(out)])

    arr = np.load(out)
    assert arr.shape == (2, 4, 4)

    # ACGT → channel-first eye matrix
    np.testing.assert_array_equal(arr[0], np.eye(4, dtype=arr.dtype))
    # TGCA
    tgca = np.array(
        [
            [0, 0, 0, 1],  # A
            [0, 0, 1, 0],  # C
            [0, 1, 0, 0],  # G
            [1, 0, 0, 0],  # T
        ],
        dtype=arr.dtype,
    )
    np.testing.assert_array_equal(arr[1], tgca)

    es = ExogeneousSequences(str(fasta))
    try:
        raw = es.get_all_region_one_hot()
        np.testing.assert_array_equal(arr, raw.transpose(0, 2, 1))
    finally:
        es.close()


def test_onehot_rejects_mixed_lengths(tmp_path: Path):
    """SPEC020: mixed-length sequences → ValueError."""
    fasta = tmp_path / "mixed.fa"
    _write_fasta(
        fasta,
        [
            ("seq1", "ACGT"),
            ("seq2", "ACG"),
        ],
    )
    with pytest.raises(ValueError, match="length-homogeneous"):
        _run(
            [
                "onehot",
                "--fasta",
                str(fasta),
                "--opath",
                str(tmp_path / "oh.npy"),
            ]
        )


# ---------------------------------------------------------------------------
# motif_search
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _meme_fixture().is_file(), reason=_SKIP_NO_MEME)
def test_motif_search_writes_npy_per_motif_shape_n_l(tmp_path: Path):
    """SPEC020: one `<output_header>.<motif_name>.npy` per motif, shape (N, L)."""
    meme = _meme_fixture()
    mm = MemeMotif(str(meme))
    names = mm.get_motif_list()
    max_w = max(mm.get_motif_length(n) for n in names)
    seq_len = max(max_w + 4, 8)

    if TINY_FA.is_file() and meme == TINY_MEME:
        fasta = TINY_FA
        n_seq = 2
        seq_len = 8
    else:
        fasta = tmp_path / "seqs.fa"
        pad = ("ACGT" * ((seq_len // 4) + 2))[:seq_len]
        alt = ("TGCA" * ((seq_len // 4) + 2))[:seq_len]
        _write_fasta(fasta, [("seq1", pad), ("seq2", alt)])
        n_seq = 2

    header = tmp_path / "ms"
    _run(
        [
            "motif_search",
            "--fasta",
            str(fasta),
            "--motif_file",
            str(meme),
            "--output_header",
            str(header),
            "--estimate_background_freq",
            "False",
            "--reverse_complement",
            "False",
        ]
    )

    for name in names:
        path = Path(f"{header}.{name}.npy")
        assert path.is_file(), f"missing {path}"
        track = np.load(path)
        assert track.shape == (n_seq, seq_len)


# ---------------------------------------------------------------------------
# Optional: argparse flag wiring
# ---------------------------------------------------------------------------


def test_motif_search_argparse_bool_flag_defaults_and_wiring():
    """SPEC020: estimate_background_freq / reverse_complement via str2bool."""
    parser = argparse.ArgumentParser()
    Motif.set_parser_motif_search(parser)

    defaults = parser.parse_args(
        ["--fasta", "x.fa", "--motif_file", "m.meme", "--output_header", "out"]
    )
    assert defaults.estimate_background_freq is True
    assert defaults.reverse_complement is False

    flipped = parser.parse_args(
        [
            "--fasta",
            "x.fa",
            "--motif_file",
            "m.meme",
            "--output_header",
            "out",
            "--estimate_background_freq",
            "False",
            "--reverse_complement",
            "True",
        ]
    )
    assert flipped.estimate_background_freq is False
    assert flipped.reverse_complement is True
