"""SPEC024 contract tests: MotifTools anti_motif command."""

from __future__ import annotations

import os
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


def test_spec024_anti_motif_writes_meme_file(tmp_path):
    """Path output writes a complete MEME collection derived from the source file."""
    out = tmp_path / "anti.meme"
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert out.is_file()
    result = MemeMotif(str(out))
    assert result.get_motif_list() == ["anti_SPEC_TINY"]


def test_spec024_anti_motif_stdout_only_data(tmp_path):
    """--output - writes MEME data to stdout and keeps diagnostics off stdout."""
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        "-",
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith("MEME version")
    assert "anti_SPEC_TINY" in completed.stdout
    meme_path = tmp_path / "stdout.meme"
    meme_path.write_text(completed.stdout, encoding="utf-8")
    parsed = MemeMotif(str(meme_path))
    assert parsed.get_motif_list() == ["anti_SPEC_TINY"]


def test_spec024_anti_motif_rejects_force_with_stdout():
    """--output - cannot be combined with --force (SPEC024)."""
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        "-",
        "--force",
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr + completed.stdout
    assert completed.stdout == ""


def test_spec024_anti_motif_refuses_overwrite_without_force(tmp_path):
    """Existing destination paths are protected unless --force is supplied."""
    out = tmp_path / "anti.meme"
    out.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    assert out.read_text(encoding="utf-8") == "placeholder"
    assert "Traceback" not in completed.stderr


def test_spec024_anti_motif_force_overwrites_destination(tmp_path):
    """--force replaces an existing destination after successful completion."""
    out = tmp_path / "anti.meme"
    out.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        str(out),
        "--force",
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert "placeholder" not in out.read_text(encoding="utf-8")
    assert MemeMotif(str(out)).get_motif_list() == ["anti_SPEC_TINY"]


def test_spec024_anti_motif_rejects_missing_parent_directory(tmp_path):
    """Missing output parent directories fail before writing (SPEC024)."""
    out = tmp_path / "missing" / "anti.meme"
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    assert not out.exists()
    assert "Traceback" not in completed.stderr


def test_spec024_anti_motif_invalid_motif_file_exits_2(tmp_path):
    """Preflight MEME validation failures exit 2 without traceback."""
    bad = tmp_path / "bad.meme"
    bad.write_text("not a meme file\n", encoding="utf-8")
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(bad),
        "--output",
        str(tmp_path / "out.meme"),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "out.meme").exists()


def test_spec024_anti_motif_success_is_silent_on_path_output(tmp_path):
    """Successful path output produces no stdout/stderr (SPEC024)."""
    out = tmp_path / "anti.meme"
    completed = _run_motiftools(
        "anti_motif",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
