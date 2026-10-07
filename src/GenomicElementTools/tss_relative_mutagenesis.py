"""TSS-relative mutagenesis for GenomicElementTools."""

from __future__ import annotations

import csv
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from RGTools.ExogenousSequences import ExogenousSequences
from RGTools.GenomicElements import GenomicElements
from RGTools.TSSRelativeMutagenesis import (
    MutationRegion,
    MutationRound,
    MutationTarget,
    compute_tss_relative_mutagenesis,
    MutationSequence,
    validate_targets,
    validate_region_chromosomes,
    validate_round_id,
    validate_output_orientation,
)

MANIFEST_COLUMNS = ("round_id", "coordinate_stat", "target_fasta", "strand")
OUTPUT_MANIFEST_COLUMNS = (
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
)

@dataclass(frozen=True)
class RoundSpec:
    round_id: str
    coordinate_stat: Path
    target_fasta: Path
    strand: str
    round_index: int  # one-based


def _resolve_path(raw: str, manifest_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _load_rounds(manifest_path: Path) -> list[RoundSpec]:
    if not manifest_path.is_file():
        raise ValueError(f"Round manifest not found: {manifest_path}")

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Round manifest is empty.")
        if tuple(reader.fieldnames) != MANIFEST_COLUMNS:
            raise ValueError(
                "Round manifest header must be exactly "
                f"{list(MANIFEST_COLUMNS)}; found {list(reader.fieldnames)}."
            )
        rows = list(reader)

    if not rows:
        raise ValueError("Round manifest must contain at least one round.")

    manifest_dir = manifest_path.parent
    rounds: list[RoundSpec] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        round_id = row["round_id"]
        if not round_id:
            raise ValueError(f"Round {index}: round_id must be nonempty.")
        if round_id in seen_ids:
            raise ValueError(f"Duplicate round_id {round_id!r}.")
        validate_round_id(round_id, round_index=index)
        seen_ids.add(round_id)

        strand = row["strand"]
        if strand not in ("+", "-"):
            raise ValueError(
                f"Round {round_id}: strand must be '+' or '-', found {strand!r}."
            )

        coord_path = _resolve_path(row["coordinate_stat"], manifest_dir)
        target_path = _resolve_path(row["target_fasta"], manifest_dir)
        if not coord_path.is_file():
            raise ValueError(
                f"Round {round_id}: coordinate_stat not found: {coord_path}"
            )
        if not target_path.is_file():
            raise ValueError(
                f"Round {round_id}: target_fasta not found: {target_path}"
            )

        rounds.append(
            RoundSpec(
                round_id=round_id,
                coordinate_stat=coord_path,
                target_fasta=target_path,
                strand=strand,
                round_index=index,
            )
        )
    return rounds


def _load_targets(path: Path, *, round_id: str) -> list[MutationTarget]:
    es = ExogenousSequences(str(path))
    ids = list(es.get_sequence_ids())
    seqs = list(es.get_all_region_seqs())
    if not ids:
        raise ValueError(f"Round {round_id}: target FASTA is empty ({path}).")

    records: list[MutationTarget] = []
    for target_id, seq in zip(ids, seqs):
        records.append(MutationTarget(target_id=target_id, sequence=seq))
    return list(validate_targets(records, round_id=round_id))


def _load_coordinate_stat(path: Path, *, n_regions: int, round_id: str) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if len(loaded.files) != 1:
            raise ValueError(
                f"Round {round_id}: coordinate NPZ must contain exactly one array."
            )
        arr = loaded[loaded.files[0]]
    else:
        arr = loaded
    if arr.shape != (n_regions, 1):
        raise ValueError(
            f"Round {round_id}: coordinate_stat must have shape ({n_regions}, 1); "
            f"found {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"Round {round_id}: coordinate_stat must be integer-valued; "
            f"found dtype {arr.dtype}."
        )
    return np.asarray(arr, dtype=np.int64)


def _staging_dir(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.staging"


def _backup_dir(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.bak"


def _detect_remnants(output_dir: Path) -> list[Path]:
    found: list[Path] = []
    for remnant in (_staging_dir(output_dir), _backup_dir(output_dir)):
        if remnant.exists():
            found.append(remnant)
    # Unique crash leftovers from older unique naming, if any.
    parent = output_dir.parent
    prefix_staging = f".{output_dir.name}.staging."
    prefix_bak = f".{output_dir.name}.bak."
    if parent.exists():
        for child in parent.iterdir():
            name = child.name
            if name.startswith(prefix_staging) or name.startswith(prefix_bak):
                found.append(child)
    return found


def _write_bundle(
    bundle_dir: Path,
    *,
    sequences: tuple[MutationSequence, ...],
    event_rows: list[dict[str, str]],
    replaced_windows: dict[str, list[tuple[str, str]]] | None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)
    fasta_path = bundle_dir / "sequences.fasta"
    with fasta_path.open("w") as handle:
        for item in sequences:
            handle.write(f">{item.sequence_id}\n{item.sequence}\n")

    manifest_path = bundle_dir / "manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OUTPUT_MANIFEST_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in event_rows:
            writer.writerow(row)

    if replaced_windows is not None:
        replaced_dir = bundle_dir / "replaced"
        replaced_dir.mkdir()
        for round_id, records in replaced_windows.items():
            path = replaced_dir / f"{round_id}.fasta"
            with path.open("w") as handle:
                for seq_id, seq in records:
                    handle.write(f">{seq_id}\n{seq}\n")


def _publish_bundle(
    *,
    output_dir: Path,
    sequences: tuple[MutationSequence, ...],
    event_rows: list[dict[str, str]],
    replaced_windows: dict[str, list[tuple[str, str]]] | None,
    force: bool,
) -> None:
    parent = output_dir.parent
    if not parent.exists():
        raise OSError(f"Output parent directory does not exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise OSError(f"Output parent directory is not writable: {parent}")

    remnants = _detect_remnants(output_dir)
    if remnants:
        names = ", ".join(str(path) for path in remnants)
        raise OSError(
            "Interrupted publication remnants detected beside the output "
            f"directory; resolve manually before retrying: {names}"
        )

    if output_dir.exists() and not force:
        raise OSError(
            f"Refusing to overwrite existing output directory: {output_dir} "
            "(use --force to replace it)"
        )

    # Unique staging sibling for this invocation.
    token = uuid.uuid4().hex
    staging = parent / f".{output_dir.name}.staging.{token}"
    backup = parent / f".{output_dir.name}.bak.{token}"
    try:
        _write_bundle(
            staging,
            sequences=sequences,
            event_rows=event_rows,
            replaced_windows=replaced_windows,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    backed_up = False
    try:
        if output_dir.exists():
            os.replace(output_dir, backup)
            backed_up = True
        os.replace(staging, output_dir)
    except Exception:
        if backed_up and backup.exists():
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(backup, output_dir)
        if staging.exists():
            shutil.rmtree(staging)
        raise

    if backup.exists():
        shutil.rmtree(backup)


class TssRelativeMutagenesis:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        GenomicElements.set_parser_genome(parser)
        parser.add_argument(
            "--round_manifest",
            help=(
                "TSV round manifest with header "
                "round_id, coordinate_stat, target_fasta, strand."
            ),
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_dir",
            help="Output directory bundle for sequences.fasta and manifest.tsv.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--write_replaced_windows",
            help="Also write replaced/<round_id>.fasta audit FASTAs.",
            action="store_true",
        )
        parser.add_argument(
            "--force",
            help="Replace an existing output directory after successful staging.",
            action="store_true",
        )
        parser.add_argument(
            "--output_orientation",
            help=(
                "Orientation of final sequences.fasta records: "
                "'genomic' (default, genomic-forward) or 'strand' "
                "(transcriptional orientation from the unique round strand)."
            ),
            choices=["genomic", "strand"],
            default="genomic",
        )

    @staticmethod
    def main(args):
        if args.region_file_type != "TREbed":
            raise ValueError(
                "tss_relative_mutagenesis requires region_file_type 'TREbed'; "
                f"got {args.region_file_type!r}."
            )

        output_dir = Path(args.output_dir).resolve()
        force = bool(args.force)
        write_replaced = bool(args.write_replaced_windows)

        # Detect remnants before expensive work.
        parent = output_dir.parent
        if parent.exists():
            remnants = _detect_remnants(output_dir)
            if remnants:
                names = ", ".join(str(path) for path in remnants)
                raise OSError(
                    "Interrupted publication remnants detected beside the output "
                    f"directory; resolve manually before retrying: {names}"
                )
        if output_dir.exists() and not force:
            raise OSError(
                f"Refusing to overwrite existing output directory: {output_dir} "
                "(use --force to replace it)"
            )
        if not parent.exists():
            raise OSError(f"Output parent directory does not exist: {parent}")

        ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=args.fasta_path,
        )
        bed = ge.get_region_bed_table()
        n_regions = ge.get_num_regions()
        region_seqs = ge.get_all_region_seqs()
        regions = list(bed.iter_regions())

        computation_regions = tuple(
            MutationRegion(
                region_row=index,
                chrom=str(region["chrom"]),
                start=int(region["start"]),
                end=int(region["end"]),
                region_name=str(region["name"]),
                fwd_tss=region["fwdTSS"],
                rev_tss=region["revTSS"],
                sequence=region_seq,
            )
            for index, (region, region_seq) in enumerate(zip(regions, region_seqs), start=1)
        )
        validate_region_chromosomes(computation_regions)

        rounds = _load_rounds(Path(args.round_manifest).resolve())
        output_orientation = args.output_orientation
        validate_output_orientation(rounds, output_orientation)

        round_targets: list[list[MutationTarget]] = []
        round_coords: list[np.ndarray] = []
        for round_spec in rounds:
            targets = _load_targets(round_spec.target_fasta, round_id=round_spec.round_id)
            coords = _load_coordinate_stat(
                round_spec.coordinate_stat,
                n_regions=n_regions,
                round_id=round_spec.round_id,
            )
            round_targets.append(targets)
            round_coords.append(coords)

        computation_rounds = tuple(
            MutationRound(
                round_id=round_spec.round_id,
                strand=round_spec.strand,
                coordinates=coords,
                targets=tuple(MutationTarget(target_id=t.target_id, sequence=t.sequence) for t in targets),
                round_index=round_spec.round_index,
            )
            for round_spec, targets, coords in zip(rounds, round_targets, round_coords)
        )
        result = compute_tss_relative_mutagenesis(
            computation_regions,
            computation_rounds,
            output_orientation=output_orientation,
            write_replaced_windows=write_replaced,
        )
        event_rows = [
            {key: str(value) for key, value in asdict(event).items()}
            for event in result.events
        ]
        replaced_windows = None if result.replaced_windows is None else {
            round_id: list(records) for round_id, records in result.replaced_windows.items()
        }

        _publish_bundle(
            output_dir=output_dir,
            sequences=result.sequences,
            event_rows=event_rows,
            replaced_windows=replaced_windows,
            force=force,
        )
