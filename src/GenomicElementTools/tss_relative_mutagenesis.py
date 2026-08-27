"""TSS-relative mutagenesis for GenomicElementTools."""

from __future__ import annotations

import csv
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Data.IUPACData import ambiguous_dna_complement, ambiguous_dna_values

from RGTools.ExogenousSequences import ExogenousSequences
from RGTools.GenomicElements import GenomicElements
from RGTools.TSSRelativeCoordinates import tss_relative_to_track_index

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

_IUPAC_DNA = frozenset(ambiguous_dna_values.keys()) | frozenset(
    b.lower() for b in ambiguous_dna_values.keys()
)
_ROUND_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RoundSpec:
    round_id: str
    coordinate_stat: Path
    target_fasta: Path
    strand: str
    round_index: int  # one-based


@dataclass(frozen=True)
class TargetRecord:
    target_id: str
    sequence: str


@dataclass
class DerivedSequence:
    region_row: int  # one-based
    chrom: str
    start: int
    end: int
    region_name: str
    target_id: str
    sequence: str

    @property
    def sequence_id(self) -> str:
        width = max(6, len(str(self.region_row)))
        return (
            f"r{self.region_row:0{width}d}|"
            f"{self.chrom}:{self.start}-{self.end}|"
            f"target={self.target_id}"
        )


def _resolve_path(raw: str, manifest_dir: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _validate_iupac_dna(seq: str, *, context: str) -> None:
    if not seq:
        raise ValueError(f"Empty mutation target rejected ({context}).")
    bad = sorted({base for base in seq if base not in _IUPAC_DNA})
    if bad:
        raise ValueError(
            f"Non-IUPAC DNA characters {bad} in mutation target ({context})."
        )


def _reverse_complement_iupac(seq: str) -> str:
    """Reverse-complement IUPAC DNA while preserving input case."""
    complement_map = {
        **ambiguous_dna_complement,
        **{k.lower(): v.lower() for k, v in ambiguous_dna_complement.items()},
    }
    # BioPython maps use uppercase only; fill lowercase explicitly.
    try:
        return "".join(complement_map[base] for base in reversed(seq))
    except KeyError as exc:
        raise ValueError(
            f"Cannot reverse-complement non-IUPAC base {exc.args[0]!r}."
        ) from exc


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
        if not _ROUND_ID_SAFE.match(round_id):
            raise ValueError(
                f"Round {index}: round_id {round_id!r} is not safe as a "
                "filename component (use letters, digits, '.', '_', '-')."
            )
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


def _load_targets(path: Path, *, round_id: str) -> list[TargetRecord]:
    es = ExogenousSequences(str(path))
    ids = list(es.get_sequence_ids())
    seqs = list(es.get_all_region_seqs())
    if not ids:
        raise ValueError(f"Round {round_id}: target FASTA is empty ({path}).")

    seen: set[str] = set()
    records: list[TargetRecord] = []
    for target_id, seq in zip(ids, seqs):
        if not target_id:
            raise ValueError(f"Round {round_id}: target ID must be nonempty.")
        if any(ch.isspace() for ch in target_id):
            raise ValueError(
                f"Round {round_id}: target ID {target_id!r} contains whitespace."
            )
        if "|" in target_id:
            raise ValueError(
                f"Round {round_id}: target ID {target_id!r} contains reserved "
                "delimiter '|'."
            )
        if target_id in seen:
            raise ValueError(
                f"Round {round_id}: duplicate target ID {target_id!r}."
            )
        seen.add(target_id)
        _validate_iupac_dna(seq, context=f"round {round_id}, target {target_id}")
        records.append(TargetRecord(target_id=target_id, sequence=seq))

    lengths = {len(rec.sequence) for rec in records}
    if len(lengths) != 1:
        raise ValueError(
            f"Round {round_id}: all targets in a round must have equal length; "
            f"found lengths {sorted(lengths)}."
        )
    return records


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
    derived: list[DerivedSequence],
    event_rows: list[dict[str, str]],
    replaced_windows: dict[str, list[tuple[str, str]]] | None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)
    fasta_path = bundle_dir / "sequences.fasta"
    with fasta_path.open("w") as handle:
        for item in derived:
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
    derived: list[DerivedSequence],
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
            derived=derived,
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

        for region in regions:
            if "|" in str(region["chrom"]):
                raise ValueError(
                    f"Chromosome {region['chrom']!r} contains reserved delimiter '|'."
                )

        rounds = _load_rounds(Path(args.round_manifest).resolve())
        round_targets: list[list[TargetRecord]] = []
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

        first_ids = [rec.target_id for rec in round_targets[0]]
        first_id_set = set(first_ids)
        for round_spec, targets in zip(rounds[1:], round_targets[1:]):
            later_ids = {rec.target_id for rec in targets}
            if later_ids != first_id_set:
                raise ValueError(
                    f"Round {round_spec.round_id}: target ID set "
                    f"{sorted(later_ids)} must match first-round set "
                    f"{sorted(first_id_set)}."
                )

        # Expand region-major × first-round target order.
        derived: list[DerivedSequence] = []
        for region_index, (region, region_seq) in enumerate(
            zip(regions, region_seqs), start=1
        ):
            for target in round_targets[0]:
                derived.append(
                    DerivedSequence(
                        region_row=region_index,
                        chrom=str(region["chrom"]),
                        start=int(region["start"]),
                        end=int(region["end"]),
                        region_name=str(region["name"]),
                        target_id=target.target_id,
                        sequence=region_seq,
                    )
                )

        event_rows: list[dict[str, str]] = []
        replaced_windows: dict[str, list[tuple[str, str]]] | None = (
            {round_spec.round_id: [] for round_spec in rounds} if write_replaced else None
        )

        # Preflight every placement bound before mutating.
        for round_spec, targets, coords in zip(rounds, round_targets, round_coords):
            target_length = len(targets[0].sequence)
            tss_field = "fwdTSS" if round_spec.strand == "+" else "revTSS"
            for item in derived:
                region = regions[item.region_row - 1]
                start = int(region["start"])
                end = int(region["end"])
                selected_tss = int(region[tss_field])
                coord = int(coords[item.region_row - 1, 0])
                context = (
                    f"round={round_spec.round_id}, region_row={item.region_row}, "
                    f"target={item.target_id}, coord={coord}"
                )
                if coord == 0:
                    raise ValueError(
                        f"TSS-relative coordinate zero is invalid ({context})."
                    )
                if selected_tss == -1:
                    raise ValueError(
                        f"Selected {tss_field}=-1 is missing ({context})."
                    )
                if not (start <= selected_tss < end):
                    raise ValueError(
                        f"Selected {tss_field}={selected_tss} is outside interval "
                        f"[{start}, {end}) ({context})."
                    )
                try:
                    tss_relative_to_track_index(
                        strand=round_spec.strand,
                        coord=coord,
                        start=start,
                        end=end,
                        tss=selected_tss,
                        track_window_size=target_length,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Replacement window does not fit inside region "
                        f"[{start}, {end}) ({context})."
                    ) from exc

        # Apply rounds sequentially.
        for round_spec, targets, coords in zip(rounds, round_targets, round_coords):
            target_by_id = {rec.target_id: rec for rec in targets}
            target_length = len(targets[0].sequence)
            tss_field = "fwdTSS" if round_spec.strand == "+" else "revTSS"

            for item in derived:
                region = regions[item.region_row - 1]
                start = int(region["start"])
                end = int(region["end"])
                selected_tss = int(region[tss_field])
                coord = int(coords[item.region_row - 1, 0])
                target_rec = target_by_id[item.target_id]
                insert_seq = target_rec.sequence
                if round_spec.strand == "-":
                    insert_seq = _reverse_complement_iupac(insert_seq)

                offset = tss_relative_to_track_index(
                    strand=round_spec.strand,
                    coord=coord,
                    start=start,
                    end=end,
                    tss=selected_tss,
                    track_window_size=target_length,
                )
                replaced = item.sequence[offset : offset + target_length]
                if write_replaced and replaced_windows is not None:
                    replaced_windows[round_spec.round_id].append(
                        (item.sequence_id, replaced)
                    )
                item.sequence = (
                    item.sequence[:offset]
                    + insert_seq
                    + item.sequence[offset + target_length :]
                )
                if len(item.sequence) != (end - start):
                    raise RuntimeError(
                        "Internal error: sequence length changed during mutagenesis."
                    )

                event_rows.append(
                    {
                        "sequence_id": item.sequence_id,
                        "region_row": str(item.region_row),
                        "chrom": item.chrom,
                        "start": str(item.start),
                        "end": str(item.end),
                        "region_name": item.region_name,
                        "target_id": item.target_id,
                        "round_index": str(round_spec.round_index),
                        "round_id": round_spec.round_id,
                        "strand": round_spec.strand,
                        "tss_relative_coordinate": str(coord),
                        "target_length": str(target_length),
                    }
                )

        # Long-form manifest order: derived-sequence order, then round order.
        event_rows.sort(
            key=lambda row: (
                int(row["region_row"]),
                first_ids.index(row["target_id"]),
                int(row["round_index"]),
            )
        )

        _publish_bundle(
            output_dir=output_dir,
            derived=derived,
            event_rows=event_rows,
            replaced_windows=replaced_windows,
            force=force,
        )
