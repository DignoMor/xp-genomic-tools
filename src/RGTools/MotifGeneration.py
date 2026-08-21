"""Motif generation and transformation algorithms for RGTools."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from .MemeMotif import MemeMotif, _BACKGROUND_SUM_ATOL, _PWM_ROW_ATOL


@dataclass(frozen=True)
class MotifExclusion:
    """Immutable motif name and score cutoff for candidate rejection."""

    motif_name: str
    cutoff: float


class SequenceGenerationExhaustedError(Exception):
    """Raised when constrained generation cannot satisfy all requested outputs."""

    def __init__(
        self,
        requested_count: int,
        completed_count: int,
        attempt_limit: int,
        exclusions: tuple[MotifExclusion, ...],
    ) -> None:
        self.requested_count = requested_count
        self.completed_count = completed_count
        self.attempt_limit = attempt_limit
        self.exclusions = exclusions
        exclusion_text = ", ".join(
            f"{item.motif_name}={item.cutoff}" for item in exclusions
        )
        super().__init__(
            "Sequence generation exhausted after "
            f"{attempt_limit} attempt(s) per output: completed "
            f"{completed_count} of {requested_count} requested sequence(s) "
            f"with exclusions [{exclusion_text}]."
        )


def parse_motif_exclusion(value: str) -> MotifExclusion:
    """Parse one ``MOTIF=CUTOFF`` exclusion string."""
    if not value or not value.strip():
        raise ValueError("Empty motif exclusion value.")
    if "=" not in value:
        raise ValueError(
            f"Malformed motif exclusion {value!r}; expected MOTIF=CUTOFF."
        )
    motif_name, cutoff_text = value.split("=", 1)
    if not motif_name:
        raise ValueError(
            f"Malformed motif exclusion {value!r}; motif name must not be empty."
        )
    try:
        cutoff = float(cutoff_text)
    except ValueError as exc:
        raise ValueError(
            f"Malformed motif exclusion cutoff in {value!r}; expected a finite number."
        ) from exc
    if not math.isfinite(cutoff):
        raise ValueError(
            f"Motif exclusion cutoff must be finite, found {cutoff_text!r}."
        )
    return MotifExclusion(motif_name, cutoff)


def parse_motif_exclusions(values: Sequence[str] | None) -> tuple[MotifExclusion, ...]:
    """Parse repeatable CLI exclusion values preserving argument order."""
    if not values:
        return ()
    return tuple(parse_motif_exclusion(value) for value in values)


def validate_motif_exclusions(
    meme: MemeMotif,
    exclusions: Sequence[MotifExclusion],
    *,
    alphabet: str,
    target_length: int,
    max_attempts: int | None = None,
) -> None:
    """Validate exclusion context before constrained generation begins."""
    if not exclusions:
        return
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError(
            f"max_attempts must be a positive integer, found {max_attempts}."
        )
    if alphabet != alphabet.upper():
        raise ValueError(
            "Generated alphabet must be uppercase when motif exclusions are used."
        )
    allowed = set("ACGT")
    if not set(alphabet).issubset(allowed):
        raise ValueError(
            "Generated alphabet must be drawn from ACGT when motif exclusions are used."
        )

    meme_alphabet = meme.get_alphabet()
    if meme_alphabet != "ACGT":
        raise ValueError(
            "MEME alphabet must be exactly ACGT for motif exclusions, "
            f"found {meme_alphabet!r}."
        )

    bg = np.asarray(meme.get_bg_freq(), dtype=float)
    _validate_backgrounds_for_exclusion(bg, len(meme_alphabet))

    motif_names = set(meme.get_motif_list())
    for exclusion in exclusions:
        if exclusion.motif_name not in motif_names:
            available = ", ".join(sorted(motif_names))
            raise ValueError(
                f"Unknown motif name for exclusion: {exclusion.motif_name!r}. "
                f"Available motifs: {available}."
            )
        motif_length = meme.get_motif_length(exclusion.motif_name)
        if motif_length > target_length:
            raise ValueError(
                f"Excluded motif {exclusion.motif_name!r} has length {motif_length}, "
                f"which exceeds generated length {target_length}."
            )
        MemeMotif._validate_pwm(
            np.asarray(meme.get_motif_pwm(exclusion.motif_name), dtype=float),
            exclusion.motif_name,
        )


def candidate_violates_exclusions(
    sequence: str,
    meme: MemeMotif,
    exclusions: Sequence[MotifExclusion],
) -> bool:
    """Return True when ``sequence`` matches any exclusion on either strand."""
    if not exclusions:
        return False

    alphabet = meme.get_alphabet()
    bg = meme.get_bg_freq()
    for exclusion in sorted(exclusions, key=lambda item: item.motif_name):
        pwm = meme.get_motif_pwm(exclusion.motif_name)
        scores = MemeMotif.search_one_motif(
            sequence,
            alphabet,
            pwm,
            bg,
            strand="both",
        )
        motif_length = meme.get_motif_length(exclusion.motif_name)
        valid_scores = scores[: len(sequence) - motif_length + 1]
        if float(np.max(valid_scores)) >= exclusion.cutoff:
            return True
    return False


def iter_pwm_sequences(
    meme: MemeMotif,
    motif_name: str,
    num_sequences: int,
    *,
    seed: int | None = None,
) -> Iterator[str]:
    """Sample ``num_sequences`` sequences from one motif PWM using the MEME alphabet."""
    motif_names = meme.get_motif_list()
    if not motif_names:
        raise ValueError("MEME collection must contain at least one motif.")
    if num_sequences <= 0:
        raise ValueError(
            f"num_sequences must be a positive integer, found {num_sequences}."
        )
    if motif_name not in meme.motif_info_dict:
        available = ", ".join(motif_names)
        raise ValueError(
            f"Unknown motif name {motif_name!r}. Available motifs: {available}."
        )

    alphabet = list(meme.get_alphabet())
    alphabet_length = len(alphabet)
    motif_alphabet_length = meme.get_motif_alphabet_length(motif_name)
    if motif_alphabet_length != alphabet_length:
        raise ValueError(
            f"Alphabet length mismatch for motif {motif_name!r}: collection has "
            f"{alphabet_length} symbols but motif declares alength="
            f"{motif_alphabet_length}."
        )

    pwm = np.asarray(meme.get_motif_pwm(motif_name), dtype=float)
    motif_length = meme.get_motif_length(motif_name)
    if pwm.shape != (motif_length, alphabet_length):
        raise ValueError(
            f"PWM shape mismatch for motif {motif_name!r}: expected "
            f"({motif_length}, {alphabet_length}), found {pwm.shape}."
        )
    MemeMotif._validate_pwm(pwm, motif_name)

    rng = random.Random(seed)
    for _ in range(num_sequences):
        positions = [
            rng.choices(alphabet, weights=pwm[row_index], k=1)[0]
            for row_index in range(motif_length)
        ]
        yield "".join(positions)


def iter_random_sequences(
    sequence_length: int,
    num_sequences: int,
    *,
    alphabet: str = "ACGT",
    seed: int | None = None,
    meme: MemeMotif | None = None,
    exclusions: Sequence[MotifExclusion] | None = None,
    max_attempts: int = 10000,
) -> Iterator[str]:
    """Sample fixed-length sequences uniformly from ``alphabet`` with replacement."""
    if sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be a positive integer, found {sequence_length}."
        )
    if num_sequences <= 0:
        raise ValueError(
            f"num_sequences must be a positive integer, found {num_sequences}."
        )
    if not alphabet:
        raise ValueError("alphabet must contain at least one character.")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(
            f"alphabet characters must be unique, found duplicate symbols in {alphabet!r}."
        )

    exclusion_tuple = tuple(exclusions or ())
    if exclusion_tuple:
        if meme is None:
            raise ValueError("meme is required when motif exclusions are supplied.")
        validate_motif_exclusions(
            meme,
            exclusion_tuple,
            alphabet=alphabet,
            target_length=sequence_length,
            max_attempts=max_attempts,
        )

    symbols = list(alphabet)
    rng = random.Random(seed)
    for completed in range(num_sequences):
        if not exclusion_tuple:
            yield "".join(rng.choice(symbols) for _ in range(sequence_length))
            continue

        accepted: str | None = None
        for _ in range(max_attempts):
            candidate = "".join(rng.choice(symbols) for _ in range(sequence_length))
            if not candidate_violates_exclusions(candidate, meme, exclusion_tuple):
                accepted = candidate
                break
        if accepted is None:
            raise SequenceGenerationExhaustedError(
                requested_count=num_sequences,
                completed_count=completed,
                attempt_limit=max_attempts,
                exclusions=exclusion_tuple,
            )
        yield accepted


def iter_barcodes(
    barcode_length: int,
    *,
    alphabet: str = "ACGT",
    meme: MemeMotif | None = None,
    exclusions: Sequence[MotifExclusion] | None = None,
    max_candidates: int = 1_000_000,
) -> Iterator[str]:
    """Enumerate barcode sequences in supplied-alphabet Cartesian order."""
    if barcode_length <= 0:
        raise ValueError(
            f"barcode_length must be a positive integer, found {barcode_length}."
        )
    if not alphabet:
        raise ValueError("alphabet must contain at least one character.")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError(
            f"alphabet characters must be unique, found duplicate symbols in {alphabet!r}."
        )
    if max_candidates <= 0:
        raise ValueError(
            f"max_candidates must be a positive integer, found {max_candidates}."
        )

    exclusion_tuple = tuple(exclusions or ())
    if exclusion_tuple:
        if meme is None:
            raise ValueError("meme is required when motif exclusions are supplied.")
        validate_motif_exclusions(
            meme,
            exclusion_tuple,
            alphabet=alphabet,
            target_length=barcode_length,
        )

    candidate_count = len(alphabet) ** barcode_length
    if candidate_count > max_candidates:
        raise ValueError(
            f"Barcode candidate space size {candidate_count} exceeds "
            f"max_candidates limit {max_candidates}."
        )

    symbols = list(alphabet)
    for candidate_tuple in product(symbols, repeat=barcode_length):
        candidate = "".join(candidate_tuple)
        if candidate_violates_exclusions(candidate, meme, exclusion_tuple):
            continue
        yield candidate


def make_anti_motifs(meme: MemeMotif) -> MemeMotif:
    """Derive an anti-motif collection from every motif in ``meme``.

    Applies row-wise smoothing ``normalize(PWM * nsites + 1)`` and inverse
    enrichment ``normalize(background**2 / smoothed)``. Output motif names are
    prefixed with ``anti_``. Source ``nsites`` and E-value metadata are copied
    as provenance rather than recomputed statistics. The source collection is
    never mutated.
    """
    motif_names = meme.get_motif_list()
    if not motif_names:
        raise ValueError("MEME collection must contain at least one motif.")

    bg = np.asarray(meme.get_bg_freq(), dtype=float)
    _validate_backgrounds_for_anti(bg, len(meme.get_alphabet()))

    result = meme.clone_empty()
    for name in motif_names:
        pwm = meme.get_motif_pwm(name)
        nsites = meme.get_motif_num_source_sites(name)
        source_eval = meme.get_motif_source_eval(name)
        _validate_motif_for_anti(pwm, nsites, source_eval, name)

        anti_pwm = _compute_anti_pwm(pwm, nsites, bg)
        result.add_motif(
            f"anti_{name}",
            {
                "alphabet_length": meme.get_motif_alphabet_length(name),
                "motif_length": meme.get_motif_length(name),
                "num_source_sites": nsites,
                "source_eval": source_eval,
                "pwm": anti_pwm,
            },
        )
    return result


def _validate_backgrounds_for_exclusion(bg: np.ndarray, alphabet_length: int) -> None:
    if bg.shape != (alphabet_length,):
        raise ValueError(
            "Background frequency length does not match alphabet length "
            f"for motif exclusion: expected {alphabet_length}, found {bg.shape[0]}."
        )
    if not np.all(np.isfinite(bg)):
        raise ValueError(
            "Background frequencies must be finite for motif exclusion."
        )
    if np.any(bg <= 0):
        raise ValueError(
            "Background frequencies must be positive for motif exclusion."
        )
    if not np.isclose(bg.sum(), 1.0, atol=_BACKGROUND_SUM_ATOL):
        raise ValueError(
            "Background frequencies must be normalized for motif exclusion."
        )


def _validate_backgrounds_for_anti(bg: np.ndarray, alphabet_length: int) -> None:
    if bg.shape != (alphabet_length,):
        raise ValueError(
            "Background frequency length does not match alphabet length "
            f"for anti-motif transformation: expected {alphabet_length}, "
            f"found {bg.shape[0]}."
        )
    if not np.all(np.isfinite(bg)):
        raise ValueError(
            "Background frequencies must be finite for anti-motif transformation."
        )
    if np.any(bg <= 0):
        raise ValueError(
            "Background frequencies must be positive for anti-motif transformation."
        )
    if not np.isclose(bg.sum(), 1.0, atol=_BACKGROUND_SUM_ATOL):
        raise ValueError(
            "Background frequencies must be normalized for anti-motif transformation."
        )


def _validate_motif_for_anti(
    pwm: np.ndarray,
    nsites: int,
    source_eval: float,
    motif_name: str,
) -> None:
    if nsites < 0:
        raise ValueError(
            f"Invalid nsites metadata for motif {motif_name!r}: {nsites}."
        )
    if not np.isfinite(source_eval):
        raise ValueError(
            f"Invalid non-finite E-value metadata for motif {motif_name!r}."
        )
    MemeMotif._validate_pwm(np.asarray(pwm, dtype=float), motif_name)


def _compute_anti_pwm(pwm: np.ndarray, nsites: int, bg: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pwm, dtype=float)
    smoothed = matrix * nsites + 1.0
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    inverse = (bg ** 2) / smoothed
    anti = inverse / inverse.sum(axis=1, keepdims=True)
    return anti.copy()
