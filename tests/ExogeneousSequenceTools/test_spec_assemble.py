"""SPEC017 contract tests: ExogeneousSequenceTools assemble operations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from ExogeneousSequenceTools import ExogeneousSequenceTools
from ExogeneousSequenceTools.ExogeneousSequenceAssemble import ExogeneousSequenceAssemble
from RGTools.ExogeneousSequences import ExogeneousSequences


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    ExogeneousSequenceTools.set_parser(parser)
    args = parser.parse_args(argv)
    ExogeneousSequenceTools.main(args)


def _write_fasta(path: Path, seq_ids: list[str], seqs: list[str]) -> Path:
    ExogeneousSequences.write_sequences_to_fasta(seq_ids, seqs, str(path))
    return path


def _read_fasta(path: Path) -> tuple[list[str], list[str]]:
    es = ExogeneousSequences(str(path))
    return es.get_sequence_ids(), es.get_all_region_seqs()


# ---------------------------------------------------------------------------
# add_adapter
# ---------------------------------------------------------------------------


def test_add_adapter_left_and_right_preserves_ids(tmp_path: Path):
    """left + seq + right; input IDs preserved (SPEC017)."""
    fasta = _write_fasta(tmp_path / "in.fa", ["s1", "s2"], ["AAA", "GGG"])
    left = _write_fasta(tmp_path / "left.fa", ["L"], ["TT"])
    right = _write_fasta(tmp_path / "right.fa", ["R"], ["CC"])
    out = tmp_path / "out.fa"

    _run_cli(
        [
            "assemble",
            "add_adapter",
            "--fasta",
            str(fasta),
            "--left_adapter_fasta",
            str(left),
            "--right_adapter_fasta",
            str(right),
            "--output_fasta",
            str(out),
        ]
    )

    ids, seqs = _read_fasta(out)
    assert ids == ["s1", "s2"]
    assert seqs == ["TTAAACC", "TTGGGCC"]


def test_add_adapter_multi_sequence_adapter_raises(tmp_path: Path):
    """Adapter FASTA with ≠1 sequence → ValueError (SPEC017)."""
    fasta = _write_fasta(tmp_path / "in.fa", ["s1"], ["AAA"])
    left = _write_fasta(tmp_path / "left.fa", ["L1", "L2"], ["TT", "GG"])
    out = tmp_path / "out.fa"

    with pytest.raises(ValueError):
        _run_cli(
            [
                "assemble",
                "add_adapter",
                "--fasta",
                str(fasta),
                "--left_adapter_fasta",
                str(left),
                "--output_fasta",
                str(out),
            ]
        )


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


def test_concat_id_method_5_3(tmp_path: Path):
    """id_method 5_3 yields <id5>_<id3> and seq5+seq3 (SPEC017)."""
    fasta5 = _write_fasta(tmp_path / "f5.fa", ["a", "b"], ["AA", "TT"])
    fasta3 = _write_fasta(tmp_path / "f3.fa", ["x", "y"], ["GG", "CC"])
    out = tmp_path / "out.fa"

    _run_cli(
        [
            "assemble",
            "concat",
            "--fasta5",
            str(fasta5),
            "--fasta3",
            str(fasta3),
            "--id_method",
            "5_3",
            "--output_fasta",
            str(out),
        ]
    )

    ids, seqs = _read_fasta(out)
    assert ids == ["a_x", "b_y"]
    assert seqs == ["AAGG", "TTCC"]


# ---------------------------------------------------------------------------
# barcode
# ---------------------------------------------------------------------------


def test_barcode_writes_fasta_and_metadata_csv(tmp_path: Path):
    """Writes FASTA and metadata CSV with required columns (SPEC017)."""
    barcodes = _write_fasta(
        tmp_path / "bc.fa", ["bc1", "bc2"], ["AT", "GC"]
    )
    input_fa = _write_fasta(
        tmp_path / "elems.fa", ["e1", "e2"], ["AAA", "TTT"]
    )
    out_fa = tmp_path / "out.fa"
    meta = tmp_path / "meta.csv"

    _run_cli(
        [
            "assemble",
            "barcode",
            "--barcode_fasta",
            str(barcodes),
            "--input_fasta",
            str(input_fa),
            "--input_class",
            "clsA",
            "--output_fasta",
            str(out_fa),
            "--metadata_path",
            str(meta),
            "--barcode_method",
            "5_3",
        ]
    )

    ids, seqs = _read_fasta(out_fa)
    assert ids == ["e1", "e2"]
    assert seqs == ["ATAAAAT", "GCTTTGC"]

    df = pd.read_csv(meta)
    assert list(df.columns) == ["barcode", "class", "elem_id", "elem_seq"]
    assert df["barcode"].tolist() == ["AT", "GC"]
    assert df["class"].tolist() == ["clsA", "clsA"]
    assert df["elem_id"].tolist() == ["e1", "e2"]
    assert df["elem_seq"].tolist() == seqs


def test_barcode_too_many_elements_raises(tmp_path: Path):
    """More elements than barcodes → ValueError (SPEC017)."""
    barcodes = _write_fasta(tmp_path / "bc.fa", ["bc1"], ["AT"])
    input_fa = _write_fasta(
        tmp_path / "elems.fa", ["e1", "e2"], ["AAA", "TTT"]
    )
    out_fa = tmp_path / "out.fa"
    meta = tmp_path / "meta.csv"

    with pytest.raises(ValueError):
        ExogeneousSequenceAssemble.main(
            argparse.Namespace(
                operation="barcode",
                barcode_fasta=str(barcodes),
                input_fasta=[str(input_fa)],
                input_class=["clsA"],
                output_fasta=str(out_fa),
                metadata_path=str(meta),
                barcode_method="5_3",
                fasta_id_type="original",
            )
        )
