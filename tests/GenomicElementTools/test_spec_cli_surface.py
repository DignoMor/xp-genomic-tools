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
    "tss_relative_mutagenesis",
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


def test_select_tss_relative_track_exposes_force_and_defaults():
    """select_tss_relative_track advertises --force and SPEC026 defaults."""
    action = _subparsers_action(_build_parser())
    parser = action.choices["select_tss_relative_track"]
    by_dest = {a.dest: a for a in parser._actions}

    assert "force" in by_dest
    assert by_dest["force"].option_strings == ["--force"]
    assert by_dest["relaxation"].default == 0
    assert by_dest["track_window_size"].default == 1
    assert by_dest["strand"].choices == ["+", "-"]
    option_strings = {
        flag for a in parser._actions for flag in a.option_strings
    }
    assert {
        "--track_npy",
        "--strand",
        "--target_coord",
        "--relaxation",
        "--min_score",
        "--track_window_size",
        "--coordinate_opath",
        "--mask_opath",
        "--force",
    }.issubset(option_strings)


def test_tss_relative_mutagenesis_exposes_bundle_flags():
    """tss_relative_mutagenesis advertises SPEC027 bundle and force flags."""
    action = _subparsers_action(_build_parser())
    parser = action.choices["tss_relative_mutagenesis"]
    by_dest = {a.dest: a for a in parser._actions}
    assert by_dest["write_replaced_windows"].option_strings == [
        "--write_replaced_windows"
    ]
    assert by_dest["force"].option_strings == ["--force"]
    assert by_dest["output_orientation"].option_strings == ["--output_orientation"]
    assert by_dest["output_orientation"].choices == ["genomic", "strand"]
    assert by_dest["output_orientation"].default == "genomic"
    option_strings = {
        flag for a in parser._actions for flag in a.option_strings
    }
    assert {
        "--fasta_path",
        "--region_file_path",
        "--region_file_type",
        "--round_manifest",
        "--output_dir",
        "--write_replaced_windows",
        "--force",
        "--output_orientation",
    }.issubset(option_strings)


def test_export_exogenous_sequences_exposes_orientation_and_record_id():
    """export ExogenousSequences advertises orientation and record-id modes (SPEC014 / #13)."""
    action = _subparsers_action(_build_parser())
    export_action = None
    for sub in action.choices["export"]._actions:
        if isinstance(sub, argparse._SubParsersAction):
            export_action = sub
            break
    assert export_action is not None
    parser = export_action.choices["ExogenousSequences"]
    by_dest = {a.dest: a for a in parser._actions}
    assert by_dest["output_orientation"].choices == ["genomic", "strand"]
    assert by_dest["output_orientation"].default == "genomic"
    assert by_dest["record_id"].choices == ["coordinate", "name"]
    assert by_dest["record_id"].default == "coordinate"
