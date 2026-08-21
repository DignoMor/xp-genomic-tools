"""Motif generation and transformation algorithms for RGTools."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np

from .MemeMotif import MemeMotif, _BACKGROUND_SUM_ATOL, _PWM_ROW_ATOL


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

    symbols = list(alphabet)
    rng = random.Random(seed)
    for _ in range(num_sequences):
        yield "".join(rng.choice(symbols) for _ in range(sequence_length))


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
