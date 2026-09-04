"""SPEC010–015/026–027: custom region schemas on generic GenomicElementTools CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from GenomicElementTools import GenomicElementTools
from RGTools.GenomicElements import GenomicElements

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "spec"
TINY_FA = FIXTURES / "tiny.fa"
CODE_ROOT = Path(__file__).resolve().parents[2]

# Commands whose primary region input uses the shared GE region helper.
GENERIC_PRIMARY_PARSER_PATHS = (
    ("pad_region",),
    ("count_single_bw",),
    ("count_paired_bw",),
    ("bed2tssbed",),
    ("onehot",),
    ("motif_search",),
    ("track2tss_bed",),
    ("filter_motif_score",),
    ("mask_op", "intersect"),
    ("mask_op", "union"),
    ("mask_op", "opposite"),
    ("get_context_ge", "nearest"),
    ("get_context_ge", "windowed_argmax"),
    ("export", "MaskedGE"),
    ("export", "ExogenousSequences"),
    ("export", "stat_list"),
    ("export", "CountTable"),
    ("export", "Heatmap"),
    ("export", "ChromFilteredGE"),
    ("export", "TREbed"),
    ("export", "WTES"),
    ("export", "allele_expanded_ES"),
    ("import", "stat_list"),
    ("select_tss_relative_track",),
    ("tss_relative_mutagenesis",),
)

NAMED_FORMATS = tuple(GenomicElements.get_region_file_suffix2class_dict().keys())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="GenomicElementTools")
    GenomicElementTools.set_parser(parser)
    return parser


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("subparsers action not registered")


def _parser_at(path: tuple[str, ...]) -> argparse.ArgumentParser:
    node = _build_parser()
    for name in path:
        action = _subparsers_action(node)
        node = action.choices[name]
    return node


def _write_schema(path: Path, *, base_type: str, extra_columns: list) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_type": base_type,
                "extra_columns": extra_columns,
            }
        )
    )
    return path


def _write_bed(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return path


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _read_bed_lines(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Parser inventory (SPEC010)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", GENERIC_PRIMARY_PARSER_PATHS)
def test_spec010_generic_primary_requires_named_or_custom_schema(path):
    """Generic primary region inputs require exactly one selector flag (SPEC010)."""
    parser = _parser_at(path)
    option_strings = {flag for a in parser._actions for flag in a.option_strings}
    assert "--region_file_path" in option_strings
    assert "--region_file_type" in option_strings
    assert "--region_file_schema" in option_strings

    type_action = next(
        a for a in parser._actions if "--region_file_type" in a.option_strings
    )
    schema_action = next(
        a for a in parser._actions if "--region_file_schema" in a.option_strings
    )
    assert type_action.choices == list(NAMED_FORMATS) or set(type_action.choices) == set(
        NAMED_FORMATS
    )
    assert schema_action.choices is None
    assert type_action.dest == schema_action.dest == "region_file_type"

    mex_groups = [
        g
        for g in getattr(parser, "_mutually_exclusive_groups", [])
        if type_action in g._group_actions and schema_action in g._group_actions
    ]
    assert len(mex_groups) == 1
    assert mex_groups[0].required is True

    help_text = parser.format_help()
    assert "--region_file_type" in help_text
    assert "--region_file_schema" in help_text
    assert "named" in help_text.lower() or "predefined" in help_text.lower()
    assert "schema" in help_text.lower()
    # Paths must not appear as named argparse choices.
    assert "{" + ",".join(NAMED_FORMATS) + "}" in help_text.replace(" ", "") or all(
        name in help_text for name in NAMED_FORMATS
    )


def test_spec010_shared_parser_rejects_both_selectors():
    """Mutual exclusion rejects providing both selector flags (SPEC010)."""
    parser = _parser_at(("pad_region",))
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--region_file_path",
                "regions.bed",
                "--region_file_type",
                "bed3",
                "--region_file_schema",
                "schema.json",
                "--upstream_pad",
                "1",
                "--downstream_pad",
                "1",
                "--opath",
                "out.bed",
            ]
        )


def test_spec010_shared_parser_rejects_missing_selector():
    """Exactly one of named format or custom schema is required (SPEC010)."""
    parser = _parser_at(("pad_region",))
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--region_file_path",
                "regions.bed",
                "--upstream_pad",
                "1",
                "--downstream_pad",
                "1",
                "--opath",
                "out.bed",
            ]
        )


def test_spec010_shared_parser_accepts_custom_schema_path():
    """--region_file_schema maps to the constructor selector destination (SPEC010)."""
    parser = _parser_at(("pad_region",))
    args = parser.parse_args(
        [
            "--region_file_path",
            "regions.bed",
            "--region_file_schema",
            "./custom.json",
            "--upstream_pad",
            "1",
            "--downstream_pad",
            "1",
            "--opath",
            "out.bed",
        ]
    )
    assert args.region_file_type == "./custom.json"


def test_spec010_named_format_choice_still_accepted():
    """Existing named-format invocations remain parseable (SPEC010)."""
    parser = _parser_at(("pad_region",))
    args = parser.parse_args(
        [
            "--region_file_path",
            "regions.bed",
            "--region_file_type",
            "bed3",
            "--upstream_pad",
            "1",
            "--downstream_pad",
            "1",
            "--opath",
            "out.bed",
        ]
    )
    assert args.region_file_type == "bed3"


# ---------------------------------------------------------------------------
# Behavioral CLI seams (SPEC011/012/014/026)
# ---------------------------------------------------------------------------


def test_spec011_pad_region_preserves_custom_bed3plus_metadata(tmp_path: Path):
    """Region-preserving pad keeps ordered custom fields and row order (SPEC011)."""
    schema = _write_schema(
        tmp_path / "meta.json",
        base_type="bed3",
        extra_columns=[
            {"name": "label", "dtype": "str"},
            {"name": "count", "dtype": "int"},
            {"name": "weight", "dtype": "float"},
        ],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        [
            "chrB\t10\t20\talpha\t3\t1.5",
            "chrA\t4\t8\tbeta\t.\t2.25",
        ],
    )
    out = tmp_path / "padded.bed"
    _run_cli(
        [
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--upstream_pad",
            "2",
            "--downstream_pad",
            "1",
            "--ignore_strand",
            "true",
            "--opath",
            str(out),
        ]
    )
    rows = _read_bed_lines(out)
    assert rows == [
        ["chrB", "8", "21", "alpha", "3", "1.5"],
        ["chrA", "2", "9", "beta", ".", "2.25"],
    ]


def test_spec011_bed2tssbed_center_preserves_custom_extras(tmp_path: Path):
    """Point transform retaining the region-table contract keeps extras (SPEC011)."""
    schema = _write_schema(
        tmp_path / "meta.json",
        base_type="bed6",
        extra_columns=[{"name": "note", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        [
            "chrA\t0\t4\tn1\t1.0\t+\tkeep-me",
            "chrA\t2\t8\tn2\t2.0\t-\talso",
        ],
    )
    out = tmp_path / "points.bed"
    _run_cli(
        [
            "bed2tssbed",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--output_site",
            "center",
            "--opath",
            str(out),
        ]
    )
    rows = _read_bed_lines(out)
    assert rows == [
        ["chrA", "2", "3", "n1", "1.0", "+", "keep-me"],
        ["chrA", "5", "6", "n2", "2.0", "-", "also"],
    ]


def test_spec012_count_paired_bw_uses_custom_bed6_strand_capability(tmp_path: Path):
    """Strand-aware counting uses BED6 capability, not selector spelling (SPEC012)."""
    pyBigWig = pytest.importorskip("pyBigWig")

    def _write_constant_bw(path: Path, value: float) -> Path:
        bw = pyBigWig.open(str(path), "w")
        bw.addHeader([("chrA", 8)])
        bw.addEntries(["chrA"], [0], ends=[8], values=[float(value)])
        bw.close()
        return path

    schema = _write_schema(
        tmp_path / "stranded.json",
        base_type="bed6",
        extra_columns=[{"name": "tag", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        [
            "chrA\t0\t4\ta\t1\t+\tx",
            "chrA\t0\t4\tb\t1\t-\ty",
            "chrA\t0\t4\tc\t1\t.\tz",
        ],
    )
    pl = _write_constant_bw(tmp_path / "pl.bw", 10.0)
    mn = _write_constant_bw(tmp_path / "mn.bw", 2.0)
    out = tmp_path / "counts.npy"
    _run_cli(
        [
            "count_paired_bw",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--bw_pl",
            str(pl),
            "--bw_mn",
            str(mn),
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--opath",
            str(out),
        ]
    )
    # Same windows as named bed6: + → 40; − → −8; . → 32
    np.testing.assert_allclose(np.load(out).ravel(), [40.0, -8.0, 32.0])


def test_spec012_count_paired_bw_custom_bed3plus_uses_unstranded_dot(tmp_path: Path):
    """Custom BED3+ lacks strand capability and counts as unstranded (SPEC012)."""
    pyBigWig = pytest.importorskip("pyBigWig")
    bw_path = tmp_path / "pl.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chrA", 8)])
    bw.addEntries(["chrA"], [0], ends=[8], values=[10.0])
    bw.close()
    mn_path = tmp_path / "mn.bw"
    bw = pyBigWig.open(str(mn_path), "w")
    bw.addHeader([("chrA", 8)])
    bw.addEntries(["chrA"], [0], ends=[8], values=[2.0])
    bw.close()

    schema = _write_schema(
        tmp_path / "plain.json",
        base_type="bed3",
        extra_columns=[{"name": "label", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrA\t0\t4\tone", "chrA\t0\t4\ttwo"],
    )
    out = tmp_path / "counts.npy"
    _run_cli(
        [
            "count_paired_bw",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--bw_pl",
            str(bw_path),
            "--bw_mn",
            str(mn_path),
            "--negative_mn",
            "True",
            "--flip_mn",
            "False",
            "--opath",
            str(out),
        ]
    )
    np.testing.assert_allclose(np.load(out).ravel(), [32.0, 32.0])


def test_spec013_filter_motif_score_preserves_custom_schema(tmp_path: Path):
    """Filtering retains custom fields, dtypes, and sidecar alignment (SPEC013)."""
    schema = _write_schema(
        tmp_path / "meta.json",
        base_type="bed3",
        extra_columns=[
            {"name": "label", "dtype": "str"},
            {"name": "rank", "dtype": "int"},
        ],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        [
            "chrA\t0\t2\tkeep\t1",
            "chrA\t2\t4\tdrop\t2",
            "chrA\t4\t6\tkeep2\t3",
        ],
    )
    motif = np.array([[0.0, 5.0], [0.0, 0.5], [0.0, 9.0]], dtype=float)
    motif_path = tmp_path / "motif.npy"
    np.save(motif_path, motif)
    header = tmp_path / "filtered"
    _run_cli(
        [
            "filter_motif_score",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--motif_search_npy",
            str(motif_path),
            "--output_header",
            str(header),
            "--filter_base",
            "1",
            "--min_score",
            "1.0",
        ]
    )
    out_bed = Path(str(header) + ".bed")
    rows = _read_bed_lines(out_bed)
    assert rows == [
        ["chrA", "0", "2", "keep", "1"],
        ["chrA", "4", "6", "keep2", "3"],
    ]
    filtered_motif = np.load(str(header) + ".motif.npy")
    assert filtered_motif.shape[0] == 2
    np.testing.assert_allclose(filtered_motif[:, 1], [5.0, 9.0])


def test_spec014_fixed_schema_export_does_not_carry_custom_extras(tmp_path: Path):
    """Fixed-schema TREbed export emits only its declared columns (SPEC014)."""
    schema = _write_schema(
        tmp_path / "meta.json",
        base_type="bed3",
        extra_columns=[{"name": "extra", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrA\t0\t4\tmeta-a", "chrA\t2\t6\tmeta-b"],
    )
    pl = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]], dtype=float)
    mn = np.array([[0.0, 0.0, 2.0, 0.0], [4.0, 0.0, 0.0, 0.0]], dtype=float)
    pl_path = tmp_path / "pl.npy"
    mn_path = tmp_path / "mn.npy"
    np.save(pl_path, pl)
    np.save(mn_path, mn)
    out = tmp_path / "out.TREbed"
    _run_cli(
        [
            "export",
            "TREbed",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--pl_sig_track",
            str(pl_path),
            "--mn_sig_track",
            str(mn_path),
            "--opath",
            str(out),
        ]
    )
    rows = _read_bed_lines(out)
    assert all(len(row) == 6 for row in rows)
    assert rows[0][3] == "chrA:0-4"
    assert "meta-a" not in out.read_text()


def test_spec026_trebed_only_rejects_structurally_matching_custom_schema(tmp_path: Path):
    """TREbed-only commands reject custom schemas with the same columns (SPEC026)."""
    schema = _write_schema(
        tmp_path / "tre_like.json",
        base_type="bed3",
        extra_columns=[
            {"name": "name", "dtype": "str"},
            {"name": "fwdTSS", "dtype": "int"},
            {"name": "revTSS", "dtype": "int"},
        ],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrA\t0\t4\tr1\t1\t2"],
    )
    track = tmp_path / "track.npy"
    np.save(track, np.zeros((1, 4), dtype=float))
    coord = tmp_path / "coord.npy"
    mask = tmp_path / "mask.npy"
    with pytest.raises(ValueError, match="TREbed"):
        _run_cli(
            [
                "select_tss_relative_track",
                "--region_file_path",
                str(bed),
                "--region_file_schema",
                str(schema),
                "--track_npy",
                str(track),
                "--strand",
                "+",
                "--target_coord",
                "1",
                "--min_score",
                "0",
                "--coordinate_opath",
                str(coord),
                "--mask_opath",
                str(mask),
            ]
        )
    assert not coord.exists()
    assert not mask.exists()


def test_spec011_invalid_schema_fails_before_output_created(tmp_path: Path):
    """Invalid schema fails before the destination is created (SPEC011)."""
    schema = tmp_path / "bad.json"
    schema.write_text('{"schema_version": 2, "base_type": "bed3", "extra_columns": []}')
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4"])
    out = tmp_path / "should_not_exist.bed"
    with pytest.raises(ValueError, match="schema_version"):
        _run_cli(
            [
                "pad_region",
                "--region_file_path",
                str(bed),
                "--region_file_schema",
                str(schema),
                "--upstream_pad",
                "1",
                "--downstream_pad",
                "1",
                "--ignore_strand",
                "true",
                "--opath",
                str(out),
            ]
        )
    assert not out.exists()


def test_spec011_invalid_region_value_fails_before_output_created(tmp_path: Path):
    """Invalid region values fail before the destination is created (SPEC011)."""
    schema = _write_schema(
        tmp_path / "meta.json",
        base_type="bed3",
        extra_columns=[{"name": "count", "dtype": "int"}],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4\tnot-an-int"])
    out = tmp_path / "should_not_exist.bed"
    with pytest.raises(Exception):
        _run_cli(
            [
                "pad_region",
                "--region_file_path",
                str(bed),
                "--region_file_schema",
                str(schema),
                "--upstream_pad",
                "1",
                "--downstream_pad",
                "1",
                "--ignore_strand",
                "true",
                "--opath",
                str(out),
            ]
        )
    assert not out.exists()


def test_spec010_named_format_pad_region_regression(tmp_path: Path):
    """Named-format pad_region output remains unchanged (SPEC010/011)."""
    bed = _write_bed(tmp_path / "regions.bed3", ["chrB\t10\t20", "chrA\t50\t60"])
    out = tmp_path / "padded.bed3"
    _run_cli(
        [
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed3",
            "--upstream_pad",
            "2",
            "--downstream_pad",
            "3",
            "--ignore_strand",
            "true",
            "--opath",
            str(out),
        ]
    )
    rows = _read_bed_lines(out)
    assert [r[0] for r in rows] == ["chrB", "chrA"]
    assert rows[0][1:3] == ["8", "23"]
    assert rows[1][1:3] == ["48", "63"]


# ---------------------------------------------------------------------------
# Installed console seams (SPEC010)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean_install_env(tmp_path_factory):
    """Wheel-install GenomicElementTools into an isolated venv."""
    root = tmp_path_factory.mktemp("custom_schema_cli")
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
    subprocess.run(
        [str(pip), "install", str(wheels[0])],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"scripts_dir": venv_dir / "bin", "python": venv_dir / "bin" / "python"}


def test_installed_console_pad_region_custom_schema(clean_install_env, tmp_path: Path):
    """Installed GenomicElementTools accepts --region_file_schema (SPEC010/011)."""
    schema = _write_schema(
        tmp_path / "meta.json",
        base_type="bed3",
        extra_columns=[{"name": "label", "dtype": "str"}],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t4\t8\talpha"])
    out = tmp_path / "out.bed"
    script = clean_install_env["scripts_dir"] / "GenomicElementTools"
    result = subprocess.run(
        [
            str(script),
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--upstream_pad",
            "1",
            "--downstream_pad",
            "1",
            "--ignore_strand",
            "true",
            "--opath",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert _read_bed_lines(out) == [["chrA", "3", "9", "alpha"]]


def test_installed_console_trebed_rejects_custom_schema(clean_install_env, tmp_path: Path):
    """Installed TREbed-only command rejects custom schema selectors (SPEC026)."""
    schema = _write_schema(
        tmp_path / "tre_like.json",
        base_type="bed3",
        extra_columns=[
            {"name": "name", "dtype": "str"},
            {"name": "fwdTSS", "dtype": "int"},
            {"name": "revTSS", "dtype": "int"},
        ],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4\tr1\t1\t2"])
    track = tmp_path / "track.npy"
    np.save(track, np.zeros((1, 4), dtype=float))
    script = clean_install_env["scripts_dir"] / "GenomicElementTools"
    result = subprocess.run(
        [
            str(script),
            "select_tss_relative_track",
            "--region_file_path",
            str(bed),
            "--region_file_schema",
            str(schema),
            "--track_npy",
            str(track),
            "--strand",
            "+",
            "--target_coord",
            "1",
            "--min_score",
            "0",
            "--coordinate_opath",
            str(tmp_path / "coord.npy"),
            "--mask_opath",
            str(tmp_path / "mask.npy"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "TREbed" in (result.stderr + result.stdout)


def test_installed_console_named_format_regression(clean_install_env, tmp_path: Path):
    """Installed named-format pad_region remains accepted (SPEC010)."""
    bed = _write_bed(tmp_path / "regions.bed3", ["chrA\t4\t8"])
    out = tmp_path / "out.bed3"
    script = clean_install_env["scripts_dir"] / "GenomicElementTools"
    result = subprocess.run(
        [
            str(script),
            "pad_region",
            "--region_file_path",
            str(bed),
            "--region_file_type",
            "bed3",
            "--upstream_pad",
            "1",
            "--downstream_pad",
            "1",
            "--ignore_strand",
            "true",
            "--opath",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert _read_bed_lines(out) == [["chrA", "3", "9"]]
