"""TSS-relative coordinate arithmetic for RGTools.

No-zero TSS-relative coordinates skip zero when offsetting across the TSS.
``+1`` is the selected TSS base; positive positions are downstream and negative
positions are upstream.
"""

from __future__ import annotations

from collections.abc import Iterator


def _to_linear(coord: int) -> int:
    """Map a no-zero TSS-relative coordinate onto a contiguous integer line."""
    if coord == 0:
        raise ValueError("TSS-relative coordinate zero is invalid.")
    return coord - 1 if coord > 0 else coord


def _from_linear(n: int) -> int:
    """Map a contiguous linear index back to a no-zero TSS-relative coordinate."""
    return n + 1 if n >= 0 else n


def offset_tss_relative_coordinate(coord: int, delta: int) -> int:
    """Offset a no-zero TSS-relative coordinate, skipping zero across the TSS."""
    if coord == 0:
        raise ValueError("TSS-relative coordinate zero is invalid.")
    return _from_linear(_to_linear(coord) + int(delta))


def iter_relaxed_window(target: int, relaxation: int) -> Iterator[int]:
    """Yield ascending TSS-relative coordinates in a symmetric relaxed window.

    Ticket 01 delivers only ``relaxation == 0`` (the exact target).
    """
    if target == 0:
        raise ValueError("TSS-relative coordinate zero is invalid.")
    if relaxation < 0:
        raise ValueError(
            f"relaxation must be a nonnegative integer, found {relaxation}."
        )
    if relaxation != 0:
        raise ValueError(
            "Nonzero relaxation is not yet delivered in this release slice."
        )
    yield target


def tss_relative_to_track_index(
    *,
    strand: str,
    coord: int,
    start: int,
    end: int,
    tss: int,
    track_window_size: int = 1,
) -> int:
    """Convert a TSS-relative coordinate to a row-local genomic-forward track index.

    Ticket 01 delivers exact plus-strand conversion with ``track_window_size == 1``.
    """
    if strand != "+":
        raise ValueError(
            f"Strand {strand!r} conversion is not yet delivered in this release slice; "
            "only '+' is supported."
        )
    if track_window_size < 1:
        raise ValueError(
            f"track_window_size must be a positive integer, found {track_window_size}."
        )
    if track_window_size != 1:
        raise ValueError(
            f"track_window_size={track_window_size} is not yet delivered in this "
            "release slice; only track_window_size=1 is supported."
        )
    if coord == 0:
        raise ValueError("TSS-relative coordinate zero is invalid.")
    if end <= start:
        raise ValueError(
            f"Invalid interval [{start}, {end}): end must be greater than start."
        )

    genomic = tss + _to_linear(coord)
    track_index = genomic - start
    length = end - start
    if not (0 <= track_index < length):
        raise ValueError(
            f"Track index {track_index} is out of bounds for interval [{start}, {end}) "
            f"(strand={strand!r}, coord={coord}, tss={tss}, "
            f"track_window_size={track_window_size})."
        )
    return int(track_index)
