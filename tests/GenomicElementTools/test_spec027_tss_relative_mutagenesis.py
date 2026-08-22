"""SPEC027 contract tests: GenomicElementTools tss_relative_mutagenesis."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from Bio import SeqIO

from GenomicElementTools.cli import GenomicElementTools

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tss_relative" / "mutagenesis"
GENOME = FIXTURES / "genome.fa"
REGIONS = FIXTURES / "regions.trebed"
ONE_REGION = FIXTURES / "one_region.trebed"
DUP_INTERVALS = FIXTURES / "dup_intervals.trebed"

MANIFEST_HEADER = ["round_id", "coordinate_stat", "target_fasta", "strand"]
OUTPUT_MANIFEST_COLUMNS = [
    "sequence_id",
    "region_row",
    "chrom",
    "start",
    "end",
    "region_name",
    "target_id",
    "round_index",
    "round_id",
    "strand",
    "tss_relative_coordinate",
    "target_length",
]


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _write_coord(path: Path, values: list[int]) -> Path:
    arr = np.asarray(values, dtype=np.int64).reshape(-1, 1)
    np.save(path, arr)
    return path


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    with path.open("w") as handle:
        for seq_id, seq in records:
            handle.write(f">{seq_id}\n{seq}\n")
    return path


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADER, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    return [(rec.id, str(rec.seq)) for rec in SeqIO.parse(path, "fasta")]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _base_argv(
    *,
    genome: Path,
    regions: Path,
    manifest: Path,
    output_dir: Path,
    write_replaced_windows: bool = False,
    force: bool = False,
) -> list[str]:
    argv = [
        "tss_relative_mutagenesis",
        "--fasta_path",
        str(genome),
        "--region_file_path",
        str(regions),
        "--region_file_type",
        "TREbed",
        "--round_manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
    ]
    if write_replaced_windows:
        argv.append("--write_replaced_windows")
    if force:
        argv.append("--force")
    return argv


def _one_round_plus_setup(tmp_path: Path) -> tuple[Path, Path]:
    """One region, one plus round, one target AAA at +1 (len 3)."""
    work = tmp_path / "work"
    work.mkdir()
    coord = _write_coord(work / "coord.npy", [1])
    target = _write_fasta(work / "targets.fa", [("t1", "AAA")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "rA",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "+",
            }
        ],
    )
    out = tmp_path / "out"
    return manifest, out


def test_subcommand_is_discoverable():
    """GenomicElementTools registers tss_relative_mutagenesis (SPEC010/027)."""
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert "tss_relative_mutagenesis" in action.choices
    sub = action.choices["tss_relative_mutagenesis"]
    dests = {a.dest for a in sub._actions}
    assert {
        "fasta_path",
        "region_file_path",
        "region_file_type",
        "round_manifest",
        "output_dir",
    }.issubset(dests)


def test_one_round_plus_publishes_length_preserving_bundle(tmp_path: Path):
    """One-round plus replacement preserves length and publishes sequences + manifest."""
    manifest, out = _one_round_plus_setup(tmp_path)
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )

    fasta_path = out / "sequences.fasta"
    manifest_path = out / "manifest.tsv"
    assert fasta_path.is_file()
    assert manifest_path.is_file()
    assert not (out / "replaced").exists()

    records = _read_fasta(fasta_path)
    assert len(records) == 1
    seq_id, seq = records[0]
    assert seq_id == "r000001|chr1:100-110|target=t1"
    assert len(seq) == 10
    # region ACGTACGTAC; +1 len3 replaces CGT at index 5 with AAA
    assert seq == "ACGTAAAAAC"

    rows = _read_manifest(manifest_path)
    assert list(rows[0].keys()) == OUTPUT_MANIFEST_COLUMNS
    assert len(rows) == 1
    row = rows[0]
    assert row["sequence_id"] == seq_id
    assert row["region_row"] == "1"
    assert row["chrom"] == "chr1"
    assert row["start"] == "100"
    assert row["end"] == "110"
    assert row["region_name"] == "r1"
    assert row["target_id"] == "t1"
    assert row["round_index"] == "1"
    assert row["round_id"] == "rA"
    assert row["strand"] == "+"
    assert row["tss_relative_coordinate"] == "1"
    assert row["target_length"] == "3"


def test_relative_manifest_paths_resolve_from_manifest_dir(tmp_path: Path):
    """Coordinate and target paths resolve relative to the round-manifest directory."""
    nested = tmp_path / "wf" / "inputs"
    nested.mkdir(parents=True)
    _write_coord(nested / "coord.npy", [1])
    _write_fasta(nested / "targets.fa", [("t1", "GGG")])
    manifest = _write_manifest(
        nested / "rounds.tsv",
        [
            {
                "round_id": "plus1",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "+",
            }
        ],
    )
    out = tmp_path / "bundle"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )
    _, seq = _read_fasta(out / "sequences.fasta")[0]
    assert seq == "ACGTAGGGAC"


def test_n_regions_one_target_preserve_region_order(tmp_path: Path):
    """One target and N regions produce N finals in region order."""
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1, 1, 1])
    _write_fasta(work / "targets.fa", [("t1", "TTT")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "rA",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "+",
            }
        ],
    )
    out = tmp_path / "out"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=REGIONS,
            manifest=manifest,
            output_dir=out,
        )
    )
    records = _read_fasta(out / "sequences.fasta")
    assert [rid for rid, _ in records] == [
        "r000001|chr1:100-110|target=t1",
        "r000002|chr1:200-210|target=t1",
        "r000003|chr1:300-320|target=t1",
    ]
    assert all(len(seq) == (10 if i < 2 else 20) for i, (_, seq) in enumerate(records))
    assert records[0][1] == "ACGTATTTAC"
    assert records[1][1] == "ACGTATTTAC"
    # region2 width 20; +1 at fwdTSS 310 → index 10; replace GTA with TTT
    assert records[2][1] == "ACGTACGTACTTTCGTACGT"


def test_missing_output_parent_is_rejected(tmp_path: Path):
    """Missing output parent is rejected without creating artifacts."""
    manifest, _ = _one_round_plus_setup(tmp_path)
    missing_parent = tmp_path / "nope" / "out"
    with pytest.raises(OSError, match="[Pp]arent"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=missing_parent,
            )
        )
    assert not missing_parent.exists()


def test_existing_output_dir_rejected_without_force(tmp_path: Path):
    """Existing output directory is rejected without --force."""
    manifest, out = _one_round_plus_setup(tmp_path)
    out.mkdir()
    (out / "marker.txt").write_text("keep")
    with pytest.raises(OSError, match="[Ee]xist|[Rr]efus|[Ff]orce"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert (out / "marker.txt").read_text() == "keep"
    assert not (out / "sequences.fasta").exists()


# --- Ticket 02: mutation target groups ---


def test_multiple_targets_expand_region_major(tmp_path: Path):
    """N regions × M targets yield N*M finals in region-major then target order."""
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1, 1])
    _write_fasta(work / "targets.fa", [("tA", "AAA"), ("tB", "CCC")])
    regions = tmp_path / "two.trebed"
    regions.write_text(
        "chr1\t100\t110\tr1\t105\t105\n"
        "chr1\t200\t210\tr2\t205\t205\n"
    )
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "rA",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "+",
            }
        ],
    )
    out = tmp_path / "out"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=regions,
            manifest=manifest,
            output_dir=out,
        )
    )
    records = _read_fasta(out / "sequences.fasta")
    assert [rid for rid, _ in records] == [
        "r000001|chr1:100-110|target=tA",
        "r000001|chr1:100-110|target=tB",
        "r000002|chr1:200-210|target=tA",
        "r000002|chr1:200-210|target=tB",
    ]
    assert [seq for _, seq in records] == [
        "ACGTAAAAAC",
        "ACGTACCCAC",
        "ACGTAAAAAC",
        "ACGTACCCAC",
    ]
    rows = _read_manifest(out / "manifest.tsv")
    assert len(rows) == 4
    assert [row["region_row"] for row in rows] == ["1", "1", "2", "2"]
    assert [row["target_id"] for row in rows] == ["tA", "tB", "tA", "tB"]
    assert [row["round_index"] for row in rows] == ["1", "1", "1", "1"]


def test_duplicate_intervals_remain_unique_by_row(tmp_path: Path):
    """Duplicate genomic intervals stay unique via one-based region_row."""
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1, 1])
    _write_fasta(work / "targets.fa", [("t1", "GGG")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "rA",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "+",
            }
        ],
    )
    out = tmp_path / "out"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=DUP_INTERVALS,
            manifest=manifest,
            output_dir=out,
        )
    )
    ids = [rid for rid, _ in _read_fasta(out / "sequences.fasta")]
    assert ids == [
        "r000001|chr1:100-110|target=t1",
        "r000002|chr1:100-110|target=t1",
    ]
    rows = _read_manifest(out / "manifest.tsv")
    assert [row["region_name"] for row in rows] == ["a", "b"]


def test_target_id_with_reserved_delimiter_rejected(tmp_path: Path):
    """Target IDs containing '|' are rejected before output."""
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("bad|id", "AAA")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "rA",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "+",
            }
        ],
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="delimiter"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


