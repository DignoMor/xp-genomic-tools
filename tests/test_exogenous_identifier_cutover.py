"""Clean-install acceptance tests for the exogenous identifier hard cutover."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = CODE_ROOT / "tests" / "fixtures" / "spec"
TINY_FA = FIXTURES / "tiny.fa"
REGIONS_BED3 = FIXTURES / "regions.bed3"

EXPECTED_SUBCOMMANDS = {
    "assemble",
    "mutagenesis",
    "gen_track",
    "track_dim_reduction",
    "print_stat",
    "motif_search",
    "onehot",
}


@pytest.fixture(scope="module")
def clean_install_env(tmp_path_factory):
    """Build a wheel and install RGTools into an isolated venv."""
    root = tmp_path_factory.mktemp("exogenous_cutover")
    wheel_dir = root / "wheels"
    venv_dir = root / "venv"

    for stale in (CODE_ROOT / "build", CODE_ROOT / "dist"):
        if stale.exists():
            shutil.rmtree(stale)
    for egg_info in CODE_ROOT.glob("*.egg-info"):
        shutil.rmtree(egg_info)
    for egg_info in (CODE_ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(egg_info)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(CODE_ROOT),
            "-w",
            str(wheel_dir),
            "--no-deps",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1

    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    pip = venv_dir / "bin" / "pip"
    py = venv_dir / "bin" / "python"
    subprocess.run(
        [str(pip), "install", str(wheels[0])],
        check=True,
        capture_output=True,
        text=True,
    )

    return {
        "python": py,
        "scripts_dir": venv_dir / "bin",
        "site_packages": venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    }


def _run_python(env: dict, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(env["python"]), "-c", code],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Positive Python (clean install)
# ---------------------------------------------------------------------------


def test_corrected_qualified_import_and_sequence_access(clean_install_env):
    fasta = str(TINY_FA)
    result = _run_python(
        clean_install_env,
        f"""
from RGTools.ExogenousSequences import ExogenousSequences
es = ExogenousSequences({fasta!r})
ids = es.get_sequence_ids()
seqs = es.get_all_region_seqs()
assert len(ids) == 2
assert len(seqs) == 2
assert "ACGTACGT" in seqs
""",
    )
    assert result.returncode == 0, result.stderr


def test_corrected_root_export(clean_install_env):
    fasta = str(TINY_FA)
    result = _run_python(
        clean_install_env,
        f"""
from RGTools import ExogenousSequences
es = ExogenousSequences({fasta!r})
assert es.get_all_region_lens() == [8, 8]
""",
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Negative Python (clean install, subprocess)
# ---------------------------------------------------------------------------


def test_misspelled_module_import_fails(clean_install_env):
    result = _run_python(
        clean_install_env,
        "from RGTools.ExogeneousSequences import ExogeneousSequences",
    )
    assert result.returncode != 0


def test_misspelled_class_import_fails(clean_install_env):
    result = _run_python(
        clean_install_env,
        "from RGTools.ExogenousSequences import ExogeneousSequences",
    )
    assert result.returncode != 0


def test_misspelled_root_export_absent(clean_install_env):
    result = _run_python(
        clean_install_env,
        "from RGTools import ExogeneousSequences",
    )
    assert result.returncode != 0


def test_misspelled_export_method_absent(clean_install_env):
    bed3 = str(REGIONS_BED3)
    fasta = str(TINY_FA)
    result = _run_python(
        clean_install_env,
        f"""
from RGTools.GenomicElements import GenomicElements
ge = GenomicElements({bed3!r}, "bed3", {fasta!r})
assert not hasattr(ge, "export_exogeneous_sequences")
ge.close()
""",
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Positive CLI (clean install)
# ---------------------------------------------------------------------------


def test_exogenous_sequence_tools_console_help(clean_install_env):
    script = clean_install_env["scripts_dir"] / "ExogenousSequenceTools"
    result = subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in EXPECTED_SUBCOMMANDS:
        assert name in result.stdout


def test_python_m_exogenous_sequence_tools_help(clean_install_env):
    result = subprocess.run(
        [str(clean_install_env["python"]), "-m", "ExogenousSequenceTools", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in EXPECTED_SUBCOMMANDS:
        assert name in result.stdout


def test_exogenous_sequence_tools_onehot_help(clean_install_env):
    """Representative subcommand dispatch surface is reachable."""
    script = clean_install_env["scripts_dir"] / "ExogenousSequenceTools"
    result = subprocess.run(
        [str(script), "onehot", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--fasta" in result.stdout


def test_exogenous_sequence_tools_onehot_dispatch(clean_install_env, tmp_path):
    """Representative existing command succeeds via the corrected console script."""
    script = clean_install_env["scripts_dir"] / "ExogenousSequenceTools"
    opath = tmp_path / "onehot.npy"
    result = subprocess.run(
        [
            str(script),
            "onehot",
            "--fasta",
            str(TINY_FA),
            "--opath",
            str(opath),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert opath.is_file()

    module_opath = tmp_path / "onehot_module.npy"
    module_result = subprocess.run(
        [
            str(clean_install_env["python"]),
            "-m",
            "ExogenousSequenceTools",
            "onehot",
            "--fasta",
            str(TINY_FA),
            "--opath",
            str(module_opath),
        ],
        capture_output=True,
        text=True,
    )
    assert module_result.returncode == 0, module_result.stderr
    assert module_opath.is_file()


# ---------------------------------------------------------------------------
# Negative CLI (clean install)
# ---------------------------------------------------------------------------


def test_misspelled_console_script_absent(clean_install_env):
    script = clean_install_env["scripts_dir"] / "ExogeneousSequenceTools"
    assert not script.exists()

    result = _run_python(
        clean_install_env,
        """
from importlib.metadata import entry_points
scripts = {ep.name for ep in entry_points().get("console_scripts", [])}
assert "ExogeneousSequenceTools" not in scripts
assert "ExogenousSequenceTools" in scripts
""",
    )
    assert result.returncode == 0, result.stderr


def test_misspelled_module_invocation_fails(clean_install_env):
    result = subprocess.run(
        [str(clean_install_env["python"]), "-m", "ExogeneousSequenceTools", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# GenomicElementTools export token (clean install)
# ---------------------------------------------------------------------------


def test_export_exogenous_sequences_accepted(clean_install_env):
    result = subprocess.run(
        [
            str(clean_install_env["scripts_dir"] / "GenomicElementTools"),
            "export",
            "ExogenousSequences",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--opath" in result.stdout


def test_export_exogeneous_sequences_rejected(clean_install_env):
    result = subprocess.run(
        [
            str(clean_install_env["scripts_dir"] / "GenomicElementTools"),
            "export",
            "ExogeneousSequences",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
