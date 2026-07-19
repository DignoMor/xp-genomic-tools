"""SPEC018 contract tests: ExogeneousSequenceTools mutagenesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ExogeneousSequenceTools.Mutagenesis import Mutagenesis
from ExogeneousSequenceTools.cli import ExogeneousSequenceTools
from RGTools.ExogeneousSequences import ExogeneousSequences


def _write_fasta(path: Path, ids: list[str], seqs: list[str]) -> Path:
    ExogeneousSequences.write_sequences_to_fasta(ids, seqs, str(path))
    return path


def _run_mutagenesis(
    *,
    fasta: Path,
    loc_npy: Path,
    mut_fasta: Path,
    output_fasta: Path,
) -> None:
    """Invoke via nested Mutagenesis entrypoint (same path as CLI dispatcher)."""
    args = argparse.Namespace(
        fasta=str(fasta),
        loc_npy=str(loc_npy),
        mut_fasta=str(mut_fasta),
        output_fasta=str(output_fasta),
    )
    Mutagenesis.mutagenesis_main(args)


def _run_via_dispatcher(
    *,
    fasta: Path,
    loc_npy: Path,
    mut_fasta: Path,
    output_fasta: Path,
) -> None:
    """Invoke via ExogeneousSequenceTools.main (CLI dispatcher)."""
    args = argparse.Namespace(
        subcommand="mutagenesis",
        fasta=str(fasta),
        loc_npy=str(loc_npy),
        mut_fasta=str(mut_fasta),
        output_fasta=str(output_fasta),
    )
    ExogeneousSequenceTools.main(args)


def _load_output(output_fasta: Path) -> tuple[list[str], list[str]]:
    es = ExogeneousSequences(str(output_fasta))
    ids = list(es.get_region_bed_table().get_chrom_names())
    seqs = list(es.get_all_region_seqs())
    return ids, seqs


# ---------------------------------------------------------------------------
# Element-wise mode (equal N)
# ---------------------------------------------------------------------------


def test_element_wise_replacement_splice_and_ids(tmp_path: Path):
    """Equal N → pair i with i; ID <input>_mut_<target>; splice at loc."""
    fasta = _write_fasta(
        tmp_path / "input.fa",
        ["seq1", "seq2", "seq3"],
        ["ATCG", "TTGA", "CCAT"],
    )
    mut_fasta = _write_fasta(
        tmp_path / "mut.fa",
        ["mut1", "mut2", "mut3"],
        ["C", "A", "T"],
    )
    loc = np.array([[0], [1], [2]], dtype=np.int64)
    loc_npy = tmp_path / "loc.npy"
    np.save(loc_npy, loc)
    out = tmp_path / "out.fa"

    _run_mutagenesis(
        fasta=fasta, loc_npy=loc_npy, mut_fasta=mut_fasta, output_fasta=out
    )

    ids, seqs = _load_output(out)
    assert len(ids) == 3
    assert ids == ["seq1_mut_mut1", "seq2_mut_mut2", "seq3_mut_mut3"]

    # output_seq = seq[:loc] + target + seq[loc + len(target):]
    assert seqs[0] == "ATCG"[:0] + "C" + "ATCG"[0 + 1 :]  # CTCG
    assert seqs[1] == "TTGA"[:1] + "A" + "TTGA"[1 + 1 :]  # TAGA
    assert seqs[2] == "CCAT"[:2] + "T" + "CCAT"[2 + 1 :]  # CCTT
    assert seqs == ["CTCG", "TAGA", "CCTT"]


def test_element_wise_preserves_length_when_target_matches_span(tmp_path: Path):
    """Length preserved when len(target) equals replaced segment length."""
    fasta = _write_fasta(tmp_path / "input.fa", ["s0"], ["ACGTACGT"])
    mut_fasta = _write_fasta(tmp_path / "mut.fa", ["t0"], ["NN"])
    loc_npy = tmp_path / "loc.npy"
    np.save(loc_npy, np.array([[3]], dtype=np.int64))
    out = tmp_path / "out.fa"

    _run_via_dispatcher(
        fasta=fasta, loc_npy=loc_npy, mut_fasta=mut_fasta, output_fasta=out
    )

    ids, seqs = _load_output(out)
    assert ids == ["s0_mut_t0"]
    expected = "ACGTACGT"[:3] + "NN" + "ACGTACGT"[3 + 2 :]
    assert seqs == [expected]
    assert len(seqs[0]) == len("ACGTACGT")


# ---------------------------------------------------------------------------
# Broadcast mode (unequal N)
# ---------------------------------------------------------------------------


def test_broadcast_output_count_and_order(tmp_path: Path):
    """Unequal N,M → N*M outputs; outer targets, inner inputs; ID formula."""
    # N=2 inputs, M=3 targets → 6 outputs
    fasta = _write_fasta(
        tmp_path / "input.fa",
        ["inA", "inB"],
        ["AAAA", "CCCC"],
    )
    mut_fasta = _write_fasta(
        tmp_path / "mut.fa",
        ["tX", "tY", "tZ"],
        ["G", "T", "N"],
    )
    loc = np.array([[1], [2]], dtype=np.int64)
    loc_npy = tmp_path / "loc.npy"
    np.save(loc_npy, loc)
    out = tmp_path / "out.fa"

    _run_mutagenesis(
        fasta=fasta, loc_npy=loc_npy, mut_fasta=mut_fasta, output_fasta=out
    )

    ids, seqs = _load_output(out)
    assert len(ids) == 2 * 3

    # Outer loop targets, inner loop inputs
    expected_ids = [
        "inA_mut_tX",
        "inB_mut_tX",
        "inA_mut_tY",
        "inB_mut_tY",
        "inA_mut_tZ",
        "inB_mut_tZ",
    ]
    assert ids == expected_ids

    def splice(seq: str, loc_i: int, target: str) -> str:
        return seq[:loc_i] + target + seq[loc_i + len(target) :]

    expected_seqs = [
        splice("AAAA", 1, "G"),
        splice("CCCC", 2, "G"),
        splice("AAAA", 1, "T"),
        splice("CCCC", 2, "T"),
        splice("AAAA", 1, "N"),
        splice("CCCC", 2, "N"),
    ]
    assert seqs == expected_seqs


def test_broadcast_via_cli_dispatcher(tmp_path: Path):
    """Broadcast path reachable through ExogeneousSequenceTools.main."""
    fasta = _write_fasta(tmp_path / "input.fa", ["e0"], ["GGGG"])
    mut_fasta = _write_fasta(
        tmp_path / "mut.fa",
        ["m0", "m1"],
        ["A", "C"],
    )
    loc_npy = tmp_path / "loc.npy"
    np.save(loc_npy, np.array([[0]], dtype=np.int64))
    out = tmp_path / "out.fa"

    _run_via_dispatcher(
        fasta=fasta, loc_npy=loc_npy, mut_fasta=mut_fasta, output_fasta=out
    )

    ids, seqs = _load_output(out)
    assert len(ids) == 1 * 2
    assert ids == ["e0_mut_m0", "e0_mut_m1"]
    assert seqs == ["AGGG", "CGGG"]
