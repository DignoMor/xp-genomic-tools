"""SPEC023 contract tests: MotifTools barcodes command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
