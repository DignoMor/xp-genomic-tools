"""SPEC022 contract tests: MotifTools shared CLI surface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from MotifTools.cli import MotifTools

CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"

EXPECTED_SUBCOMMANDS = {
    "random_seq",
    "pwm_seq",
    "barcodes",
    "anti_motif",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="MotifTools")
    MotifTools.set_parser(parser)
    return parser


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("top-level subparsers action not registered")


def test_spec022_help_lists_expected_subcommands():
    """MotifTools --help lists the SPEC022 subcommand inventory."""
    help_text = _build_parser().format_help()
    for name in EXPECTED_SUBCOMMANDS:
        assert name in help_text


def test_spec022_parser_registers_expected_subcommands():
    """Dispatcher argparse choices match the planned subcommand set (SPEC022)."""
    action = _subparsers_action(_build_parser())
    assert action.dest == "subcommand"
    assert set(action.choices) == EXPECTED_SUBCOMMANDS
    assert action.required is True


def test_spec022_anti_motif_flags():
    """anti_motif exposes required motif/output flags and optional --force."""
    action = _subparsers_action(_build_parser())
    anti = action.choices["anti_motif"]
    dests = {a.dest for a in anti._actions}
    assert {"motif_file", "output", "force"}.issubset(dests)
    option_strings = {flag for a in anti._actions for flag in a.option_strings}
    assert {"--motif_file", "--output", "--force"}.issubset(option_strings)


def test_spec022_unknown_subcommand_raises_value_error():
    """Unknown top-level subcommand → ValueError (SPEC022)."""
    with pytest.raises(ValueError, match="[Ss]ubcommand"):
        MotifTools.main(argparse.Namespace(subcommand="not_a_real_cmd"))


def test_spec022_python_module_entrypoint_help():
    """python -m MotifTools behaves like the console entrypoint (SPEC022)."""
    completed = subprocess.run(
        [sys.executable, "-m", "MotifTools", "--help"],
        cwd=CODE_ROOT,
        env={**dict(**{"PYTHONPATH": str(SRC_ROOT)}), **dict(__import__("os").environ)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for name in EXPECTED_SUBCOMMANDS:
        assert name in completed.stdout


def test_spec022_missing_subcommand_exits_2():
    """Missing required subcommand exits 2 without traceback (SPEC022)."""
    completed = subprocess.run(
        [sys.executable, "-m", "MotifTools"],
        cwd=CODE_ROOT,
        env={**dict(**{"PYTHONPATH": str(SRC_ROOT)}), **dict(__import__("os").environ)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
