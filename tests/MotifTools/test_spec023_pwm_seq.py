"""SPEC023 contract tests: MotifTools pwm_seq command."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from RGTools import MemeMotif

CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
FIXTURES = CODE_ROOT / "tests" / "fixtures"
TINY_MEME = FIXTURES / "spec" / "tiny.meme"


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


def test_spec023_pwm_seq_writes_fasta_with_stable_ids(tmp_path):
    """Path output writes FASTA records with pwm_<motif>_<index> identifiers."""
    out = tmp_path / "pwm.fasta"
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--num_sequences",
        "2",
        "--seed",
        "42",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    text = out.read_text(encoding="utf-8")
    assert text.startswith(">pwm_SPEC_TINY_0\n")
    assert ">pwm_SPEC_TINY_1\n" in text
    assert text.endswith("\n")
    lines = text.strip().split("\n")
    assert len(lines) == 4
    assert all(len(lines[i]) == 3 for i in (1, 3))


def test_spec023_pwm_seq_seed_zero_golden(tmp_path):
    """Seed 0 reproduces stable golden FASTA bytes for pwm_seq."""
    out = tmp_path / "pwm.fasta"
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--num_sequences",
        "3",
        "--seed",
        "0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    expected = (
        ">pwm_SPEC_TINY_0\n"
        "GCG\n"
        ">pwm_SPEC_TINY_1\n"
        "ACG\n"
        ">pwm_SPEC_TINY_2\n"
        "CCG\n"
    )
    assert out.read_text(encoding="utf-8") == expected


def test_spec023_pwm_seq_stdout_only_data(tmp_path):
    """--output - writes FASTA data to stdout only."""
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
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
    assert completed.stdout == ">pwm_SPEC_TINY_0\nGCG\n"


def test_spec023_pwm_seq_rejects_force_with_stdout():
    """--output - cannot be combined with --force."""
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--num_sequences",
        "1",
        "--output",
        "-",
        "--force",
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr + completed.stdout
    assert completed.stdout == ""


def test_spec023_pwm_seq_unknown_motif_exits_2(tmp_path):
    """Unknown motif names fail before output with exit 2."""
    out = tmp_path / "pwm.fasta"
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "MISSING",
        "--num_sequences",
        "1",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr


def test_spec023_pwm_seq_rejects_nonpositive_count(tmp_path):
    """Non-positive --num_sequences fails before output."""
    out = tmp_path / "pwm.fasta"
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--num_sequences",
        "0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()


def test_spec023_pwm_seq_refuses_overwrite_without_force(tmp_path):
    """Existing destination paths are protected unless --force is supplied."""
    out = tmp_path / "pwm.fasta"
    out.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "pwm_seq",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--num_sequences",
        "1",
        "--seed",
        "0",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    assert out.read_text(encoding="utf-8") == "placeholder"
    assert "Traceback" not in completed.stderr
