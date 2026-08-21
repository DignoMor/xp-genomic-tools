"""Motif generation and transformation algorithms for RGTools."""

from __future__ import annotations

import numpy as np

from .MemeMotif import MemeMotif, _BACKGROUND_SUM_ATOL, _PWM_ROW_ATOL


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
