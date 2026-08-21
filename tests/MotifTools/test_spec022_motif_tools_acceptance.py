"""SPEC022 acceptance tests: complete MotifTools subprocess surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
FIXTURES = CODE_ROOT / "tests" / "fixtures"
TINY_MEME = FIXTURES / "spec" / "tiny.meme"

EXPECTED_COMMANDS = ("random_seq", "pwm_seq", "barcodes", "anti_motif")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    return env


def _run_module(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "MotifTools", *args],
        cwd=cwd or CODE_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_console(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    motif_tools = CODE_ROOT / ".venv/bin/MotifTools"
    if not motif_tools.is_file():
        motif_tools = Path(sys.executable).with_name("MotifTools")
    return subprocess.run(
        [str(motif_tools), *args],
        cwd=cwd or CODE_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_spec022_acceptance_help_lists_exact_command_inventory():
    """MotifTools --help exposes exactly the four delivered subcommands."""
    completed = _run_module("--help")
    assert completed.returncode == 0
    for command in EXPECTED_COMMANDS:
        assert command in completed.stdout


def test_spec022_acceptance_python_module_matches_console_help():
    """python -m MotifTools and the console script expose the same subcommands."""
    module_help = _run_module("--help")
    console_help = _run_console("--help")
    assert module_help.returncode == 0
    assert console_help.returncode == 0
    for command in EXPECTED_COMMANDS:
        assert command in module_help.stdout
        assert command in console_help.stdout


def test_spec022_acceptance_all_commands_succeed_via_module_and_console(tmp_path):
    """Each delivered command succeeds through both invocation forms."""
    cases = (
        [
            "anti_motif",
            "--motif_file",
            str(TINY_MEME),
            "--output",
            str(tmp_path / "anti.meme"),
        ],
        [
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
            str(tmp_path / "pwm.fasta"),
        ],
        [
            "random_seq",
            "--sequence_length",
            "3",
            "--num_sequences",
            "1",
            "--seed",
            "0",
            "--output",
            str(tmp_path / "random.fasta"),
        ],
        [
            "barcodes",
            "--barcode_length",
            "1",
            "--output",
            str(tmp_path / "barcodes.fasta"),
        ],
    )
    for index, argv in enumerate(cases):
        def with_output(suffix: str) -> list[str]:
            args = list(argv)
            output_index = args.index("--output") + 1
            output_path = Path(args[output_index])
            args[output_index] = str(output_path.with_name(f"{suffix}_{output_path.name}"))
            return args

        module = _run_module(*with_output("module"), cwd=tmp_path)
        console = _run_console(*with_output("console"), cwd=tmp_path)
        assert module.returncode == 0, module.stderr
        assert console.returncode == 0, console.stderr
        assert module.stdout == ""
        assert module.stderr == ""


def test_spec022_acceptance_parser_errors_exit_two_without_traceback():
    """Missing required flags exit 2 without traceback."""
    completed = _run_module("random_seq", "--sequence_length", "3")
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr


def test_spec022_acceptance_stdout_output_is_data_only(tmp_path):
    """--output - writes only FASTA/MEME data to stdout on success."""
    completed = _run_module(
        "barcodes",
        "--barcode_length",
        "1",
        "--output",
        "-",
        cwd=tmp_path,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith(">barcode_0")
