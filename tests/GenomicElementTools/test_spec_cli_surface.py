"""SPEC010 contract tests: GenomicElementTools shared CLI surface."""

from __future__ import annotations

import argparse

import pytest

from GenomicElementTools import GenomicElementTools

EXPECTED_SUBCOMMANDS = {
    "count_single_bw",
    "count_paired_bw",
    "pad_region",
    "bed2tssbed",
    "onehot",
    "motif_search",
    "track2tss_bed",
    "filter_motif_score",
    "export",
    "import",
    "get_context_ge",
    "mask_op",
    "select_tss_relative_track",
}

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="GenomicElementTools")
    GenomicElementTools.set_parser(parser)
    return parser


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("top-level subparsers action not registered")


def test_help_lists_expected_subcommands():
    """GenomicElementTools --help lists the SPEC010 subcommand inventory."""
    help_text = _build_parser().format_help()
    for name in EXPECTED_SUBCOMMANDS:
        assert name in help_text


def test_parser_registers_expected_subcommands():
    """Dispatcher argparse choices match the Planned subcommand set (SPEC010)."""
    action = _subparsers_action(_build_parser())
    assert action.dest == "subcommand"
    assert set(action.choices) == EXPECTED_SUBCOMMANDS


def test_unknown_subcommand_raises_value_error():
    """Unknown top-level subcommand → ValueError (SPEC010)."""
    with pytest.raises(ValueError, match="[Uu]nknown subcommand"):
        GenomicElementTools.main(argparse.Namespace(subcommand="not_a_real_cmd"))


def test_shared_region_args_on_pad_region():
    """Shared GE region flags appear on a sample subcommand (SPEC010)."""
    action = _subparsers_action(_build_parser())
    pad = action.choices["pad_region"]
    dests = {a.dest for a in pad._actions}
    assert "region_file_path" in dests
    assert "region_file_type" in dests

    option_strings = {
        flag for a in pad._actions for flag in a.option_strings
    }
    assert "--region_file_path" in option_strings
    assert "--region_file_type" in option_strings
