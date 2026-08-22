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


# --- Ticket 03: sequential rounds / strand / replaced windows ---


def test_later_round_joins_by_target_id_despite_reorder(tmp_path: Path):
    """Later rounds join by target ID even when FASTA order differs."""
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "c1.npy", [1])
    _write_coord(work / "c2.npy", [-1])
    _write_fasta(work / "t1.fa", [("tA", "AAA"), ("tB", "CCC")])
    # Reordered and different length for round 2
    _write_fasta(work / "t2.fa", [("tB", "GG"), ("tA", "TT")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "first",
                "coordinate_stat": "c1.npy",
                "target_fasta": "t1.fa",
                "strand": "+",
            },
            {
                "round_id": "second",
                "coordinate_stat": "c2.npy",
                "target_fasta": "t2.fa",
                "strand": "+",
            },
        ],
    )
    out = tmp_path / "out"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
            write_replaced_windows=True,
        )
    )
    records = _read_fasta(out / "sequences.fasta")
    assert [rid for rid, _ in records] == [
        "r000001|chr1:100-110|target=tA",
        "r000001|chr1:100-110|target=tB",
    ]
    # Round1 +1 len3 @idx5; round2 -1 len2:
    # genomic = tss + (-1) = 104; index = 104-100 = 4 for plus window size 2
    # After round1 tA: ACGTAAAAAC; replace idx4:6 (AA) with TT → ACGTTTAAAC
    # After round1 tB: ACGTACCCAC; replace idx4:6 (AC) with GG → ACGTGGCCAC
    assert records[0][1] == "ACGTTTAAAC"
    assert records[1][1] == "ACGTGGCCAC"

    rows = _read_manifest(out / "manifest.tsv")
    assert len(rows) == 4  # N*M*R = 1*2*2
    assert [row["round_id"] for row in rows] == [
        "first",
        "second",
        "first",
        "second",
    ]

    replaced_first = _read_fasta(out / "replaced" / "first.fasta")
    replaced_second = _read_fasta(out / "replaced" / "second.fasta")
    assert [rid for rid, _ in replaced_first] == [rid for rid, _ in records]
    assert [seq for _, seq in replaced_first] == ["CGT", "CGT"]
    # Second round replaced windows include first-round bases where they overlap
    assert [seq for _, seq in replaced_second] == ["AA", "AC"]
    assert all(len(seq) == 2 for _, seq in replaced_second)


def test_minus_strand_reverse_complements_iupac_target(tmp_path: Path):
    """Minus rounds use revTSS and reverse-complement IUPAC targets."""
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("t1", "ATy")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "minus1",
                "coordinate_stat": "coord.npy",
                "target_fasta": "targets.fa",
                "strand": "-",
            }
        ],
    )
    out = tmp_path / "out"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )
    _, seq = _read_fasta(out / "sequences.fasta")[0]
    # RC(ATy)=rAT; minus +1 len3 → index = 105-100-(3-1)=3; replace TAC with rAT
    assert seq == "ACGrATGTAC"
    row = _read_manifest(out / "manifest.tsv")[0]
    assert row["strand"] == "-"
    assert row["target_length"] == "3"


# --- Ticket 04: preflight ---


def test_coordinate_zero_rejected_before_output(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [0])
    _write_fasta(work / "targets.fa", [("t1", "AAA")])
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
    with pytest.raises(ValueError, match="zero"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_missing_selected_tss_rejected(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("t1", "AAA")])
    regions = tmp_path / "missing_tss.trebed"
    regions.write_text("chr1\t100\t110\tr1\t-1\t105\n")
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
    with pytest.raises(ValueError, match="fwdTSS"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=regions,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_window_overflow_rejected(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    # +1 with length 6 from index 5 overflows end at 10
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("t1", "AAAAAA")])
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
    with pytest.raises(ValueError, match="fit|bounds|window"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_empty_manifest_rejected(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    manifest = work / "rounds.tsv"
    manifest.write_text("round_id\tcoordinate_stat\ttarget_fasta\tstrand\n")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="at least one round"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_unequal_same_round_target_lengths_rejected(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("tA", "AAA"), ("tB", "CC")])
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
    with pytest.raises(ValueError, match="equal length"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_non_iupac_target_rejected(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("t1", "A.T")])
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
    with pytest.raises(ValueError, match="IUPAC"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()



# --- Ticket 04 (continued): placement / coverage preflight ---

def test_float_integral_coordinates_are_not_coerced(tmp_path: Path):
    """Float arrays are rejected even when values are integral."""
    work = tmp_path / "work"
    work.mkdir()
    path = work / "coord.npy"
    np.save(path, np.asarray([[1.0]], dtype=np.float64))
    _write_fasta(work / "targets.fa", [("t1", "AAA")])
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
    with pytest.raises(ValueError, match="integer"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_out_of_interval_selected_tss_rejected(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("t1", "AAA")])
    regions = tmp_path / "bad_tss.trebed"
    regions.write_text("chr1\t100\t110\tr1\t99\t105\n")  # fwdTSS outside
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
    with pytest.raises(ValueError, match="outside interval"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=regions,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


def test_missing_chromosome_rejected_before_output(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    _write_coord(work / "coord.npy", [1])
    _write_fasta(work / "targets.fa", [("t1", "AAA")])
    regions = tmp_path / "missing_chrom.trebed"
    regions.write_text("chrZ\t100\t110\tr1\t105\t105\n")
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
    with pytest.raises(ValueError, match="[Cc]hrom"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=regions,
                manifest=manifest,
                output_dir=out,
            )
        )
    assert not out.exists()


# --- Ticket 05: forced replacement / remnants ---


def test_force_replaces_existing_bundle(tmp_path: Path):
    """--force stages a complete replacement and swaps the output directory."""
    manifest, out = _one_round_plus_setup(tmp_path)
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )
    first = _read_fasta(out / "sequences.fasta")[0][1]

    work = tmp_path / "work"
    _write_fasta(work / "targets.fa", [("t1", "CCC")])
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
            force=True,
        )
    )
    second = _read_fasta(out / "sequences.fasta")[0][1]
    assert first == "ACGTAAAAAC"
    assert second == "ACGTACCCAC"
    assert not list(out.parent.glob(f".{out.name}.bak*"))
    assert not list(out.parent.glob(f".{out.name}.staging*"))


def test_interrupted_backup_remnant_blocks_rerun(tmp_path: Path):
    """Detected backup/staging remnants block rerun without destructive cleanup."""
    manifest, out = _one_round_plus_setup(tmp_path)
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )
    remnant = out.parent / f".{out.name}.bak"
    remnant.mkdir()
    (remnant / "old.txt").write_text("recover-me")
    with pytest.raises(OSError, match="Interrupted|remnant"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
                force=True,
            )
        )
    assert (remnant / "old.txt").read_text() == "recover-me"
    assert (out / "sequences.fasta").exists()


def test_force_does_not_remove_unrelated_sibling(tmp_path: Path):
    manifest, out = _one_round_plus_setup(tmp_path)
    sibling = tmp_path / "unrelated"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("safe")
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
            force=True,
        )
    )
    assert (sibling / "keep.txt").read_text() == "safe"


def test_force_publication_rollback_preserves_previous_bundle(tmp_path: Path, monkeypatch):
    """Ordinary failure during publication restores the previous bundle."""
    manifest, out = _one_round_plus_setup(tmp_path)
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=ONE_REGION,
            manifest=manifest,
            output_dir=out,
        )
    )
    original = (out / "sequences.fasta").read_text()
    work = tmp_path / "work"
    _write_fasta(work / "targets.fa", [("t1", "CCC")])

    import GenomicElementTools.tss_relative_mutagenesis as mut

    real_replace = mut.os.replace

    def flaky_replace(src, dst):
        src_path = Path(src)
        dst_path = Path(dst)
        # Fail when promoting a staging directory onto the destination.
        if src_path.name.startswith(f".{out.name}.staging.") and dst_path == out:
            raise OSError("induced publication failure")
        return real_replace(src, dst)

    monkeypatch.setattr(mut.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="induced publication failure"):
        _run_cli(
            _base_argv(
                genome=GENOME,
                regions=ONE_REGION,
                manifest=manifest,
                output_dir=out,
                force=True,
            )
        )
    assert (out / "sequences.fasta").read_text() == original
    assert not list(out.parent.glob(f".{out.name}.staging*"))
def test_end_to_end_selection_mask_export_mutagenesis(tmp_path: Path):
    """Select → mask_op → MaskedGE → multi-round tss_relative_mutagenesis."""
    # Use plus-capable regions (all fwdTSS set) and a track that matches row0 only.
    regions = tmp_path / "plus.trebed"
    regions.write_text(
        "chr1\t100\t110\tr1\t105\t105\n"
        "chr1\t200\t210\tr2\t205\t205\n"
    )
    track = np.zeros((2, 10), dtype=float)
    track[0, 5] = 2.0  # +1 at fwdTSS
    track[1, 5] = 0.1
    track_path = tmp_path / "track.npy"
    np.save(track_path, track)

    coord_out = tmp_path / "coord.npy"
    mask_out = tmp_path / "mask.npy"
    _run_cli(
        [
            "select_tss_relative_track",
            "--region_file_path",
            str(regions),
            "--region_file_type",
            "TREbed",
            "--track_npy",
            str(track_path),
            "--strand",
            "+",
            "--target_coord",
            "1",
            "--min_score=1.0",
            "--coordinate_opath",
            str(coord_out),
            "--mask_opath",
            str(mask_out),
        ]
    )
    assert np.array_equal(np.load(mask_out).ravel(), [True, False])

    filtered_bed = tmp_path / "filtered.trebed"
    anno_header = tmp_path / "filtered_anno"
    _run_cli(
        [
            "export",
            "MaskedGE",
            "--region_file_path",
            str(regions),
            "--region_file_type",
            "TREbed",
            "--mask_npy",
            str(mask_out),
            "--opath",
            str(filtered_bed),
            "--anno_name",
            "tss_rel_coord",
            "--anno_npy",
            str(coord_out),
            "--anno_type",
            "stat",
            "--anno_oheader",
            str(anno_header),
        ]
    )
    filtered_coord = Path(f"{anno_header}.tss_rel_coord.npy")
    assert filtered_bed.read_text().strip() == "chr1\t100\t110\tr1\t105\t105"
    assert np.load(filtered_coord).ravel().tolist() == [1]

    work = tmp_path / "mut"
    work.mkdir()
    shutil.copy(filtered_coord, work / "c1.npy")
    _write_coord(work / "c2.npy", [-1])
    _write_fasta(work / "t1.fa", [("t1", "AAA")])
    _write_fasta(work / "t2.fa", [("t1", "TT")])
    manifest = _write_manifest(
        work / "rounds.tsv",
        [
            {
                "round_id": "selected",
                "coordinate_stat": "c1.npy",
                "target_fasta": "t1.fa",
                "strand": "+",
            },
            {
                "round_id": "followup",
                "coordinate_stat": "c2.npy",
                "target_fasta": "t2.fa",
                "strand": "+",
            },
        ],
    )
    out = tmp_path / "bundle"
    _run_cli(
        _base_argv(
            genome=GENOME,
            regions=filtered_bed,
            manifest=manifest,
            output_dir=out,
            write_replaced_windows=True,
        )
    )
    records = _read_fasta(out / "sequences.fasta")
    assert len(records) == 1
    # Round1 AAA @+1 then round2 TT @-1 → ACGTTTAAAC
    assert records[0] == ("r000001|chr1:100-110|target=t1", "ACGTTTAAAC")
    rows = _read_manifest(out / "manifest.tsv")
    assert len(rows) == 2
    assert [row["round_id"] for row in rows] == ["selected", "followup"]
    assert (out / "replaced" / "selected.fasta").is_file()
    assert (out / "replaced" / "followup.fasta").is_file()


