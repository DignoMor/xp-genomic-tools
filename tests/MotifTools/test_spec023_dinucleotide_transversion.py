"""SPEC023 contract tests: MotifTools dinucleotide_transversion command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
FIXTURES = CODE_ROOT / "tests" / "fixtures"
TINY_MEME = FIXTURES / "spec" / "tiny.meme"

TINY_FASTA = ">dinucleotide_transversion_SPEC_TINY\nCAC\n"


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


def _write_meme(
    path: Path,
    name: str,
    rows: list[list[float]],
    *,
    alphabet: str = "ACGT",
    header: str | None = None,
) -> Path:
    bg = " ".join(f"{letter} 0.25" for letter in alphabet)
    matrix = "\n".join("  ".join(f"{value:.6f}" for value in row) for row in rows)
    width = len(rows)
    alength = len(alphabet)
    matrix_header = header or (
        f"letter-probability matrix: alength= {alength} w= {width} nsites= 8 E= 1e-4"
    )
    path.write_text(
        "MEME version 4\n\n"
        f"ALPHABET={alphabet}\n\n"
        "strands: + -\n\n"
        "Background letter frequencies\n"
        f"{bg}\n\n"
        f"MOTIF {name}\n"
        f"{matrix_header}\n"
        f"{matrix}\n",
        encoding="utf-8",
    )
    return path


def _write_empty_collection(path: Path) -> Path:
    path.write_text(
        "MEME version 4\n\n"
        "ALPHABET=ACGT\n\n"
        "strands: + -\n\n"
        "Background letter frequencies\n"
        "A 0.25 C 0.25 G 0.25 T 0.25\n",
        encoding="utf-8",
    )
    return path


def test_spec023_dinucleotide_transversion_tiny_odd_width_golden(tmp_path):
    """SPEC023: SPEC_TINY (w=3) yields independently specified CAC FASTA."""
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert out.read_text(encoding="utf-8") == TINY_FASTA


def test_spec023_dinucleotide_transversion_even_width_golden(tmp_path):
    """SPEC023: informative even width 2 yields independently specified CA."""
    meme = _write_meme(
        tmp_path / "even.meme",
        "EVEN2",
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ],
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "EVEN2",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_EVEN2\nCA\n"


def test_spec023_dinucleotide_transversion_width_one_golden(tmp_path):
    """SPEC023: width-one PWM yields independently specified C."""
    meme = _write_meme(tmp_path / "w1.meme", "W1", [[0.7, 0.1, 0.1, 0.1]])
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "W1",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_W1\nC\n"


def test_spec023_dinucleotide_transversion_tied_consensus_golden(tmp_path):
    """SPEC023: tied consensus columns A/C resolve by letter order to TT."""
    meme = _write_meme(
        tmp_path / "tied.meme",
        "TIED",
        [
            [0.4, 0.4, 0.1, 0.1],
            [0.4, 0.4, 0.1, 0.1],
        ],
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "TIED",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_TIED\nTT\n"


def test_spec023_dinucleotide_transversion_equal_products_golden(tmp_path):
    """SPEC023: equal minimum products resolve lexicographically to TA."""
    meme = _write_meme(
        tmp_path / "eqprod.meme",
        "EQPROD",
        [
            [0.5, 0.2, 0.2, 0.1],
            [0.2, 0.5, 0.2, 0.1],
        ],
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "EQPROD",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_EQPROD\nTA\n"


def test_spec023_dinucleotide_transversion_zero_product_ties_golden(tmp_path):
    """SPEC023: zero-probability products participate and lex-resolve to CA."""
    meme = _write_meme(
        tmp_path / "zero.meme",
        "ZERO",
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
        ],
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "ZERO",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_ZERO\nCA\n"


def test_spec023_dinucleotide_transversion_uninformative_terminal_golden(tmp_path):
    """SPEC023: uninformative odd terminal is retained; independently specified CAC."""
    meme = _write_meme(
        tmp_path / "uninf.meme",
        "UNINF",
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "UNINF",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_UNINF\nCAC\n"


def test_spec023_dinucleotide_transversion_alphabet_order_indexes_pwm(tmp_path):
    """SPEC023: PWM columns follow declared alphabet; letter order still ties."""
    meme = _write_meme(
        tmp_path / "tcga.meme",
        "TCGA_TINY",
        [
            [0.1, 0.1, 0.1, 0.7],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ],
        alphabet="TCGA",
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "TCGA_TINY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == ">dinucleotide_transversion_TCGA_TINY\nCAC\n"


def test_spec023_dinucleotide_transversion_stdout_matches_path(tmp_path):
    """SPEC023: path and stdout FASTA bytes are identical."""
    out = tmp_path / "dtv.fasta"
    path_run = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    stdout_run = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        "-",
        cwd=tmp_path,
    )
    assert path_run.returncode == 0
    assert stdout_run.returncode == 0
    assert stdout_run.stderr == ""
    assert stdout_run.stdout == TINY_FASTA
    assert out.read_text(encoding="utf-8") == stdout_run.stdout


def test_spec023_dinucleotide_transversion_repeated_runs_are_byte_identical(tmp_path):
    """SPEC023: repeated successful runs emit identical FASTA bytes."""
    first = tmp_path / "a.fasta"
    second = tmp_path / "b.fasta"
    args = (
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
    )
    run_a = _run_motiftools(*args, str(first), cwd=tmp_path)
    run_b = _run_motiftools(*args, str(second), cwd=tmp_path)
    assert run_a.returncode == 0
    assert run_b.returncode == 0
    assert first.read_bytes() == second.read_bytes() == TINY_FASTA.encode("utf-8")


def test_spec023_dinucleotide_transversion_help_lists_flags_without_seed(tmp_path):
    """SPEC023: command help lists required flags and has no seed or method selector."""
    completed = _run_motiftools("dinucleotide_transversion", "--help", cwd=tmp_path)
    assert completed.returncode == 0
    help_text = completed.stdout
    assert "--motif_file" in help_text
    assert "--motif_name" in help_text
    assert "--output" in help_text
    assert "--force" in help_text
    assert "--seed" not in help_text
    assert "--method" not in help_text
    assert "--warn_score_cutoff" not in help_text


def test_spec023_dinucleotide_transversion_missing_required_flags_exit_2(tmp_path):
    """SPEC023: missing required flags exit 2 with no traceback or target."""
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert not out.exists()


def test_spec023_dinucleotide_transversion_unknown_motif_exits_2(tmp_path):
    """SPEC023: unknown motif names fail before publication with exit 2."""
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "MISSING",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr
    assert "MISSING" in completed.stderr


def test_spec023_dinucleotide_transversion_empty_collection_exits_2(tmp_path):
    """SPEC023: empty collections fail before publication with exit 2."""
    meme = _write_empty_collection(tmp_path / "empty.meme")
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "ANY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr


def test_spec023_dinucleotide_transversion_invalid_pwm_exits_2(tmp_path):
    """SPEC023: unnormalized PWM rows fail before publication with exit 2."""
    meme = _write_meme(tmp_path / "bad.meme", "BAD", [[0.5, 0.1, 0.1, 0.1]])
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "BAD",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr


def test_spec023_dinucleotide_transversion_missing_nsites_exits_2(tmp_path):
    """SPEC023: missing required nsites/E remain rejected with exit 2."""
    meme = _write_meme(
        tmp_path / "nonsites.meme",
        "NOSITES",
        [[0.7, 0.1, 0.1, 0.1]],
        header="letter-probability matrix: alength= 4 w= 1",
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "NOSITES",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr


def test_spec023_dinucleotide_transversion_unsupported_alphabet_exits_2(tmp_path):
    """SPEC023: non-ACGT alphabets fail contextually before publication."""
    meme = _write_meme(
        tmp_path / "rna.meme",
        "RNA1",
        [[0.7, 0.1, 0.1, 0.1]],
        alphabet="ACGU",
    )
    out = tmp_path / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(meme),
        "--motif_name",
        "RNA1",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert not out.exists()
    assert "Traceback" not in completed.stderr
    assert "alphabet" in completed.stderr.lower()


def test_spec023_dinucleotide_transversion_rejects_force_with_stdout():
    """SPEC023: --output - cannot be combined with --force."""
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        "-",
        "--force",
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr + completed.stdout
    assert completed.stdout == ""


def test_spec023_dinucleotide_transversion_refuses_overwrite_without_force(tmp_path):
    """SPEC023: existing destinations are protected unless --force is supplied."""
    out = tmp_path / "dtv.fasta"
    out.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    assert out.read_text(encoding="utf-8") == "placeholder"
    assert "Traceback" not in completed.stderr


def test_spec023_dinucleotide_transversion_force_overwrites_destination(tmp_path):
    """SPEC023: --force replaces an existing destination after success."""
    out = tmp_path / "dtv.fasta"
    out.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        str(out),
        "--force",
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert out.read_text(encoding="utf-8") == TINY_FASTA


def test_spec023_dinucleotide_transversion_validation_preserves_forced_destination(tmp_path):
    """SPEC023: validation failures leave an existing forced destination unchanged."""
    out = tmp_path / "dtv.fasta"
    out.write_text("placeholder", encoding="utf-8")
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "MISSING",
        "--output",
        str(out),
        "--force",
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert out.read_text(encoding="utf-8") == "placeholder"
    assert "Traceback" not in completed.stderr


def test_spec023_dinucleotide_transversion_missing_parent_directory_exits_1(tmp_path):
    """SPEC023: missing output parent directories use the existing I/O status."""
    out = tmp_path / "missing" / "dtv.fasta"
    completed = _run_motiftools(
        "dinucleotide_transversion",
        "--motif_file",
        str(TINY_MEME),
        "--motif_name",
        "SPEC_TINY",
        "--output",
        str(out),
        cwd=tmp_path,
    )
    assert completed.returncode == 1
    assert not out.exists()
    assert "Traceback" not in completed.stderr
