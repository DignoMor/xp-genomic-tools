"""SPEC023 contract tests: MotifTools barcodes command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from RGTools import MemeMotif
from RGTools.MotifGeneration import MotifExclusion, iter_barcodes

CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
TINY_MEME = CODE_ROOT / "tests/fixtures/spec/tiny.meme"


def _run_motiftools(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "MotifTools", *args],
        cwd=cwd or CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_spec023_barcodes_nonstandard_alphabet_order(tmp_path):
    """Barcodes preserve supplied-alphabet Cartesian order in FASTA output."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes",
        "--barcode_length",
        "2",
        "--alphabet",
        "XY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert out.read_text(encoding="utf-8") == (
        ">barcode_0\n"
        "XX\n"
        ">barcode_1\n"
        "XY\n"
        ">barcode_2\n"
        "YX\n"
        ">barcode_3\n"
        "YY\n"
    )


def test_spec023_barcodes_rc_collisions_have_compact_ids(tmp_path):
    """SPEC023: RC exclusion works without motifs and preserves accepted order."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "2", "--exclude_rc_collisions",
        "--output", str(out), cwd=tmp_path,
    )
    assert completed.returncode == 0
    sequences = [line for line in out.read_text(encoding="utf-8").splitlines() if not line.startswith(">")]
    identifiers = [line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith(">")]
    assert sequences == ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "GA", "GC", "TA"]
    assert identifiers == [f">barcode_{index}" for index in range(10)]


def test_spec023_barcodes_rc_rejects_invalid_alphabet_before_publication(tmp_path):
    """SPEC023: opt-in alphabet errors exit 2 and do not create output."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "1", "--alphabet", "Ac",
        "--exclude_rc_collisions", "--output", str(out), cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "A/C/G/T" in completed.stderr
    assert not out.exists()


def test_spec023_barcodes_rc_force_preserves_existing_destination(tmp_path):
    """SPEC023: invalid RC input leaves an existing forced destination intact."""
    out = tmp_path / "barcodes.fasta"
    out.write_text("previous output\n", encoding="utf-8")
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "1", "--alphabet", "N",
        "--exclude_rc_collisions", "--output", str(out), "--force", cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert out.read_text(encoding="utf-8") == "previous output\n"


def test_spec023_barcodes_rc_stdout_contains_only_fasta(tmp_path):
    """SPEC023: standalone RC mode writes FASTA data only to stdout."""
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "1", "--exclude_rc_collisions",
        "--output", "-", cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == ">barcode_0\nA\n>barcode_1\nC\n"


def test_spec023_barcodes_rc_and_motif_exclusions_compose(tmp_path):
    """SPEC023: CLI applies motif and RC exclusions in accepted order."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "3", "--motif_file", str(TINY_MEME),
        "--exclude", "SPEC_TINY=0.0", "--exclude_rc_collisions",
        "--output", str(out), cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    motif_survivors = iter_barcodes(
        3,
        meme=MemeMotif(str(TINY_MEME)),
        exclusions=(MotifExclusion("SPEC_TINY", 0.0),),
    )
    expected = []
    accepted = set()
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    for candidate in motif_survivors:
        reverse_complement = "".join(
            complement[base] for base in reversed(candidate)
        )
        if reverse_complement not in accepted:
            expected.append(candidate)
            accepted.add(candidate)
    expected_fasta = "".join(
        f">barcode_{index}\n{candidate}\n"
        for index, candidate in enumerate(expected)
    )
    assert expected
    assert out.read_text(encoding="utf-8") == expected_fasta


def test_spec023_barcodes_rc_guard_checks_full_pre_exclusion_space(tmp_path):
    """SPEC023: RC exclusion does not reduce the candidate preflight count."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "10", "--exclude_rc_collisions",
        "--output", str(out), cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "candidate space size" in completed.stderr
    assert not out.exists()


def test_spec023_barcodes_rc_and_motif_can_write_empty_output(tmp_path):
    """SPEC023: combined filters with no survivors succeed with zero bytes."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "3", "--motif_file", str(TINY_MEME),
        "--exclude", "SPEC_TINY=-5.0", "--exclude_rc_collisions",
        "--output", str(out), cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert out.read_bytes() == b""


def test_spec023_barcodes_rc_preserves_unused_motif_file_error(tmp_path):
    """SPEC023: RC exclusion does not make a supplied unused motif file valid."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes", "--barcode_length", "1", "--motif_file", str(TINY_MEME),
        "--exclude_rc_collisions", "--output", str(out), cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "unused" in completed.stderr
    assert not out.exists()


def test_spec023_barcodes_guard_fails_before_enumeration(tmp_path):
    """Default candidate guard rejects oversized spaces without enumerating."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes",
        "--barcode_length",
        "10",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "candidate space size" in completed.stderr
    assert not out.exists()


def test_spec023_barcodes_empty_result_is_zero_byte_fasta(tmp_path):
    """No surviving barcodes succeeds silently with zero-byte FASTA output."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes",
        "--barcode_length",
        "3",
        "--motif_file",
        str(TINY_MEME),
        "--exclude",
        "SPEC_TINY=-5.0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert out.exists()
    assert out.read_bytes() == b""


def test_spec023_barcodes_stdout_only_data(tmp_path):
    """--output - writes surviving barcode FASTA to stdout only."""
    completed = _run_motiftools(
        "barcodes",
        "--barcode_length",
        "1",
        "--alphabet",
        "ACGT",
        "--output",
        "-",
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == (
        ">barcode_0\n"
        "A\n"
        ">barcode_1\n"
        "C\n"
        ">barcode_2\n"
        "G\n"
        ">barcode_3\n"
        "T\n"
    )


def test_spec023_barcodes_shared_exclusions(tmp_path):
    """Exclusion-enabled barcodes retain accepted-order identifiers."""
    out = tmp_path / "barcodes.fasta"
    completed = _run_motiftools(
        "barcodes",
        "--barcode_length",
        "3",
        "--motif_file",
        str(TINY_MEME),
        "--exclude",
        "SPEC_TINY=0.0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith(">barcode_0\nAAA\n")
    assert text.endswith(">barcode_43\nTTT\n")
    assert text.count(">barcode_") == 44
