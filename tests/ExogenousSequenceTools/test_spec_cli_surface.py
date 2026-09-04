"""SPEC016 contract tests: ExogenousSequenceTools shared CLI surface."""

from __future__ import annotations

import argparse

import pytest

from ExogenousSequenceTools import ExogenousSequenceTools

EXPECTED_SUBCOMMANDS = {
    "assemble",
    "mutagenesis",
    "gen_track",
    "track_dim_reduction",
    "print_stat",
    "motif_search",
    "onehot",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ExogenousSequenceTools")
    ExogenousSequenceTools.set_parser(parser)
    return parser


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("top-level subparsers action not registered")


def test_help_lists_expected_subcommands():
    """ExogenousSequenceTools --help lists the SPEC016 subcommand inventory."""
    help_text = _build_parser().format_help()
    for name in EXPECTED_SUBCOMMANDS:
        assert name in help_text


def test_parser_registers_expected_subcommands():
    """Dispatcher argparse choices match the Planned subcommand set (SPEC016)."""
    action = _subparsers_action(_build_parser())
    assert action.dest == "subcommand"
    assert set(action.choices) == EXPECTED_SUBCOMMANDS


def test_unknown_subcommand_raises_value_error():
    """Unknown top-level subcommand → ValueError (SPEC016)."""
    with pytest.raises(ValueError, match="[Ss]ubcommand"):
        ExogenousSequenceTools.main(argparse.Namespace(subcommand="not_a_real_cmd"))


def test_shared_fasta_arg_on_onehot():
    """Shared ES --fasta flag appears on a sample sequence subcommand (SPEC016)."""
    action = _subparsers_action(_build_parser())
    onehot = action.choices["onehot"]
    dests = {a.dest for a in onehot._actions}
    assert "fasta" in dests

    option_strings = {
        flag for a in onehot._actions for flag in a.option_strings
    }
    assert "--fasta" in option_strings
