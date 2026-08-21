"""Shared motif-exclusion wiring for MotifTools commands."""

from __future__ import annotations

from RGTools import MemeMotif
from RGTools.MotifGeneration import MotifExclusion, parse_motif_exclusions


def resolve_exclusion_inputs(
    motif_file: str | None,
    exclude_args: list[str] | None,
) -> tuple[MemeMotif | None, tuple[MotifExclusion, ...]]:
    """Load MEME input and parse repeatable exclusion flags."""
    exclusions = parse_motif_exclusions(exclude_args)
    if motif_file is None and exclusions:
        raise ValueError("Motif exclusions require --motif_file.")
    if motif_file is not None and not exclusions:
        raise ValueError(
            "Motif file supplied without exclusions is unused input; "
            "add --exclude MOTIF=CUTOFF or omit --motif_file."
        )
    if motif_file is None:
        return None, ()
    return MemeMotif(motif_file), exclusions
