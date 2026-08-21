"""SPEC023 contract tests: MotifTools random_seq command (unconstrained)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"


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


def test_spec023_random_seq_default_alphabet_seed_zero_golden(tmp_path):
    """Default ACGT alphabet with seed 0 reproduces golden FASTA bytes."""
    out = tmp_path / "random.fasta"
    completed = _run_motiftools(
        "random_seq",
        "--sequence_length",
        "4",
        "--num_sequences",
        "3",
        "--seed",
        "0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    expected = (
        ">random_seq_0\n"
        "TTAG\n"
        ">random_seq_1\n"
        "TTGT\n"
        ">random_seq_2\n"
        "GCCG\n"
    )
    assert out.read_text(encoding="utf-8") == expected


def test_spec023_random_seq_nonstandard_alphabet_order(tmp_path):
    """Alphabet symbol order is preserved in sampling."""
    out = tmp_path / "random.fasta"
    completed = _run_motiftools(
        "random_seq",
        "--sequence_length",
        "2",
        "--num_sequences",
        "2",
        "--alphabet",
        "XY",
        "--seed",
        "1",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert all(
        symbol in "XY"
        for line in text.splitlines()
        if not line.startswith(">")
        for symbol in line
    )


def test_spec023_random_seq_stdout_only_data(tmp_path):
    """--output - writes FASTA data to stdout only."""
    completed = _run_motiftools(
        "random_seq",
        "--sequence_length",
        "3",
        "--num_sequences",
        "1",
        "--seed",
        "0",
        "--output",
        "-",
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == ">random_seq_0\nTTA\n"


def test_spec023_random_seq_rejects_duplicate_alphabet_chars(tmp_path):
    """Duplicate alphabet characters fail before output."""
    out = tmp_path / "random.fasta"
    completed = _run_motiftools(
        "random_seq",
        "--sequence_length",
        "2",
        "--num_sequences",
        "1",
        "--alphabet",
        "AAC",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr


def test_spec023_random_seq_rejects_motif_file_before_output(tmp_path):
    """Motif exclusion inputs are not yet delivered."""
    out = tmp_path / "random.fasta"
    meme = tmp_path / "tiny.meme"
    meme.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "random_seq",
        "--sequence_length",
        "4",
        "--num_sequences",
        "1",
        "--motif_file",
        str(meme),
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()


def test_spec023_random_seq_rejects_exclude_before_output(tmp_path):
    """Motif exclusion flags are not yet delivered."""
    out = tmp_path / "random.fasta"
    completed = _run_motiftools(
        "random_seq",
        "--sequence_length",
        "4",
        "--num_sequences",
        "1",
        "--exclude",
        "MOTIF=1.0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
