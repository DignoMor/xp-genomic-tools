"""Internal computation for TSS-relative mutation trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from Bio.Data.IUPACData import ambiguous_dna_complement, ambiguous_dna_values

from .TSSRelativeCoordinates import tss_relative_to_track_index

_IUPAC_DNA = frozenset(ambiguous_dna_values) | frozenset(
    base.lower() for base in ambiguous_dna_values
)
_ROUND_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class MutationRegion:
    """One source region paired with its genomic-forward sequence (SPEC027)."""
    region_row: int
    chrom: str
    start: int
    end: int
    region_name: str
    fwd_tss: int
    rev_tss: int
    sequence: str


@dataclass(frozen=True)
class MutationTarget:
    """One ordered mutation target record (SPEC027)."""
    target_id: str
    sequence: str


@dataclass(frozen=True)
class MutationRound:
    """One ordered round with row-aligned integer coordinates (SPEC027)."""
    round_id: str
    strand: str
    coordinates: tuple[int, ...]
    targets: tuple[MutationTarget, ...]
    round_index: int

    def __post_init__(self) -> None:
        require_matrix = isinstance(self.coordinates, np.ndarray)
        values = np.asarray(self.coordinates)
        if values.ndim == 2:
            if values.shape[1] != 1:
                raise ValueError(
                    f"Round {self.round_id}: coordinate_stat must have shape (N, 1); "
                    f"found {values.shape}."
                )
            values = values[:, 0]
        elif require_matrix:
            raise ValueError(
                f"Round {self.round_id}: coordinate_stat must have shape (N, 1); "
                f"found {values.shape}."
            )
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(
                f"Round {self.round_id}: coordinate_stat must be a one-dimensional "
                f"integer array; found shape {values.shape}, dtype {values.dtype}."
            )
        object.__setattr__(self, "coordinates", tuple(int(value) for value in values))


@dataclass(frozen=True)
class MutationEvent:
    """One long-form mutation manifest event (SPEC027)."""
    sequence_id: str
    region_row: int
    chrom: str
    start: int
    end: int
    region_name: str
    target_id: str
    round_index: int
    round_id: str
    strand: str
    tss_relative_coordinate: int
    target_length: int

@dataclass(frozen=True)
class MutationSequence:
    """One stable-ID final sequence record (SPEC027)."""
    sequence_id: str
    region_row: int
    chrom: str
    start: int
    end: int
    region_name: str
    target_id: str
    sequence: str


@dataclass(frozen=True)
class MutationResult:
    """Complete ordered computation output consumed by CLI serialization (SPEC027)."""
    sequences: tuple[MutationSequence, ...]
    events: tuple[MutationEvent, ...]
    replaced_windows: Mapping[str, tuple[tuple[str, str], ...]] | None


def validate_output_orientation(rounds: Sequence[object], output_orientation: str) -> None:
    """Validate the CLI's early orientation rule without loading round payloads."""
    if output_orientation == "strand":
        strands = {round_spec.strand for round_spec in rounds}
        if len(strands) != 1:
            raise ValueError(
                "--output_orientation strand requires every round to declare the same "
                f"strand; found mixed strands {sorted(strands)}."
            )


def validate_region_chromosomes(regions: Sequence[MutationRegion]) -> None:
    """Reject reserved chromosome delimiters before SPEC027 computation."""
    for region in regions:
        if "|" in region.chrom:
            raise ValueError(f"Chromosome {region.chrom!r} contains reserved delimiter '|'.")


def validate_round_id(round_id: str, *, round_index: int) -> None:
    """Validate a manifest round identifier for SPEC027."""
    if not round_id or not _ROUND_ID_SAFE.fullmatch(round_id):
        raise ValueError(
            f"Round {round_index}: round_id {round_id!r} is not safe as a "
            "filename component (use letters, digits, '.', '_', '-')."
        )


def _reverse_complement_iupac(seq: str) -> str:
    complement = {
        **ambiguous_dna_complement,
        **{key.lower(): value.lower() for key, value in ambiguous_dna_complement.items()},
    }
    try:
        return "".join(complement[base] for base in reversed(seq))
    except KeyError as exc:
        raise ValueError(
            f"Cannot reverse-complement non-IUPAC base {exc.args[0]!r}."
        ) from exc


def _sequence_id(region: MutationRegion, target_id: str) -> str:
    width = max(6, len(str(region.region_row)))
    return (
        f"r{region.region_row:0{width}d}|{region.chrom}:{region.start}-{region.end}|"
        f"target={target_id}"
    )


def _validate_round(round_spec: MutationRound) -> None:
    if round_spec.strand not in ("+", "-"):
        raise ValueError(
            f"Round {round_spec.round_id}: strand must be '+' or '-', "
            f"found {round_spec.strand!r}."
        )
    validate_targets(round_spec.targets, round_id=round_spec.round_id)


def validate_targets(
    targets: Sequence[MutationTarget], *, round_id: str
) -> tuple[MutationTarget, ...]:
    """Validate ordered target records for SPEC027 loading and computation."""
    if not targets:
        raise ValueError(f"Round {round_id}: target FASTA is empty.")
    seen: set[str] = set()
    for target in targets:
        if not target.target_id:
            raise ValueError(f"Round {round_id}: target ID must be nonempty.")
        if any(ch.isspace() for ch in target.target_id):
            raise ValueError(
                f"Round {round_id}: target ID {target.target_id!r} contains whitespace."
            )
        if "|" in target.target_id:
            raise ValueError(
                f"Round {round_id}: target ID {target.target_id!r} contains reserved delimiter '|'."
            )
        if target.target_id in seen:
            raise ValueError(
                f"Round {round_id}: duplicate target ID {target.target_id!r}."
            )
        seen.add(target.target_id)
        if not target.sequence:
            raise ValueError(
                f"Empty mutation target rejected (round {round_id}, target {target.target_id})."
            )
        bad = sorted({base for base in target.sequence if base not in _IUPAC_DNA})
        if bad:
            raise ValueError(
                f"Non-IUPAC DNA characters {bad} in mutation target "
                f"(round {round_id}, target {target.target_id})."
            )
    lengths = {len(target.sequence) for target in targets}
    if len(lengths) != 1:
        raise ValueError(
            f"Round {round_id}: all targets in a round must have equal length; "
            f"found lengths {sorted(lengths)}."
        )
    return tuple(targets)


def compute_tss_relative_mutagenesis(
    regions: Sequence[MutationRegion],
    rounds: Sequence[MutationRound],
    *,
    output_orientation: str = "genomic",
    write_replaced_windows: bool = False,
) -> MutationResult:
    """Compute and validate a complete SPEC027 mutation trajectory.

    Inputs are treated as immutable: placement preflight completes before any
    sequence is changed, and the returned records own all trajectory state.
    """
    if output_orientation not in ("genomic", "strand"):
        raise ValueError(f"Unknown output orientation {output_orientation!r}.")
    if not rounds:
        raise ValueError("Round manifest must contain at least one round.")
    validate_region_chromosomes(regions)
    for position, region in enumerate(regions, start=1):
        if region.region_row != position:
            raise ValueError(
                f"Region rows must be one-based contiguous ordinals; found "
                f"{region.region_row} at position {position}."
            )
        if len(region.sequence) != region.end - region.start:
            raise ValueError(
                f"Region row {region.region_row}: sequence length {len(region.sequence)} "
                f"does not match interval length {region.end - region.start}."
            )
    seen_round_ids: set[str] = set()
    for position, round_spec in enumerate(rounds, start=1):
        if round_spec.round_index != position:
            raise ValueError(
                f"Round indexes must be one-based contiguous ordinals; found "
                f"{round_spec.round_index} at position {position}."
            )
        validate_round_id(round_spec.round_id, round_index=round_spec.round_index)
        if round_spec.round_id in seen_round_ids:
            raise ValueError(f"Duplicate round_id {round_spec.round_id!r}.")
        seen_round_ids.add(round_spec.round_id)
        _validate_round(round_spec)
        if len(round_spec.coordinates) != len(regions):
            raise ValueError(
                f"Round {round_spec.round_id}: coordinate_stat must have shape "
                f"({len(regions)}, 1); found ({len(round_spec.coordinates)}, 1)."
            )

    validate_output_orientation(rounds, output_orientation)
    if output_orientation == "strand":
        strands = {round_spec.strand for round_spec in rounds}
        output_strand = next(iter(strands))
    else:
        output_strand = None

    first_targets = rounds[0].targets
    first_ids = tuple(target.target_id for target in first_targets)
    first_id_set = set(first_ids)
    for round_spec in rounds[1:]:
        later_ids = {target.target_id for target in round_spec.targets}
        if later_ids != first_id_set:
            raise ValueError(
                f"Round {round_spec.round_id}: target ID set {sorted(later_ids)} must match "
                f"first-round set {sorted(first_id_set)}."
            )

    # Expand in source order and retain immutable source metadata separately.
    items: list[MutationSequence] = []
    for region in regions:
        for target in first_targets:
            items.append(
                MutationSequence(
                    sequence_id=_sequence_id(region, target.target_id),
                    region_row=region.region_row,
                    chrom=region.chrom,
                    start=region.start,
                    end=region.end,
                    region_name=region.region_name,
                    target_id=target.target_id,
                    sequence=region.sequence,
                )
            )

    # Placement is cached once per round and region: equal target lengths make it
    # valid for every target group, while the first target supplies diagnostics.
    placements: list[tuple[int, ...]] = []
    for round_spec in rounds:
        target_length = len(round_spec.targets[0].sequence)
        tss_field = "fwd_tss" if round_spec.strand == "+" else "rev_tss"
        round_placements: list[int] = []
        first_target = first_ids[0]
        for region, coord in zip(regions, round_spec.coordinates):
            context = (
                f"round={round_spec.round_id}, region_row={region.region_row}, "
                f"target={first_target}, coord={coord}"
            )
            if coord == 0:
                raise ValueError(f"TSS-relative coordinate zero is invalid ({context}).")
            tss = getattr(region, tss_field)
            tss_label = "fwdTSS" if round_spec.strand == "+" else "revTSS"
            if tss == -1:
                raise ValueError(f"Selected {tss_label}=-1 is missing ({context}).")
            if not (region.start <= tss < region.end):
                raise ValueError(
                    f"Selected {tss_label}={tss} is outside interval "
                    f"[{region.start}, {region.end}) ({context})."
                )
            try:
                offset = tss_relative_to_track_index(
                    strand=round_spec.strand, coord=coord, start=region.start,
                    end=region.end, tss=tss, track_window_size=target_length,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Replacement window does not fit inside region [{region.start}, {region.end}) ({context})."
                ) from exc
            round_placements.append(offset)
        placements.append(tuple(round_placements))

    mutable_sequences = [item.sequence for item in items]
    events: list[MutationEvent] = []
    replaced: dict[str, list[tuple[str, str]]] | None = (
        {round_spec.round_id: [] for round_spec in rounds} if write_replaced_windows else None
    )
    target_maps = [{target.target_id: target for target in round_spec.targets} for round_spec in rounds]
    for round_index, (round_spec, offsets, target_map) in enumerate(zip(rounds, placements, target_maps)):
        target_length = len(round_spec.targets[0].sequence)
        for item_index, item in enumerate(items):
            region_index = item.region_row - 1
            offset = offsets[region_index]
            target = target_map[item.target_id]
            insert = _reverse_complement_iupac(target.sequence) if round_spec.strand == "-" else target.sequence
            old_sequence = mutable_sequences[item_index]
            removed = old_sequence[offset : offset + target_length]
            if replaced is not None:
                replaced[round_spec.round_id].append((item.sequence_id, removed))
            mutable_sequences[item_index] = old_sequence[:offset] + insert + old_sequence[offset + target_length:]
            if len(mutable_sequences[item_index]) != item.end - item.start:
                raise RuntimeError("Internal error: sequence length changed during mutagenesis.")
            events.append(MutationEvent(
                sequence_id=item.sequence_id, region_row=item.region_row, chrom=item.chrom,
                start=item.start, end=item.end, region_name=item.region_name,
                target_id=item.target_id, round_index=round_spec.round_index,
                round_id=round_spec.round_id, strand=round_spec.strand,
                tss_relative_coordinate=round_spec.coordinates[region_index],
                target_length=target_length,
            ))

    final_sequences = []
    for item, sequence in zip(items, mutable_sequences):
        if output_strand == "-":
            sequence = _reverse_complement_iupac(sequence)
        final_sequences.append(MutationSequence(**{**item.__dict__, "sequence": sequence}))
    ordered_events = tuple(
        events[round_number * len(items) + item_number]
        for item_number in range(len(items))
        for round_number in range(len(rounds))
    )
    return MutationResult(
        sequences=tuple(final_sequences), events=ordered_events,
        replaced_windows=None if replaced is None else MappingProxyType(
            {key: tuple(value) for key, value in replaced.items()}
        ),
    )
