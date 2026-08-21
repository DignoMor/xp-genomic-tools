"""SPEC021 contract tests: RGTools.MotifGeneration public API."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from RGTools import MemeMotif
from RGTools.MotifGeneration import make_anti_motifs

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY_MEME = FIXTURES / "spec" / "tiny.meme"
SAMPLE_MEME = FIXTURES / "sample-dna-motif.meme"

_PWM_ROW_ATOL = 1e-6
_BG_SUM_ATOL = 1e-3


def _expected_anti_pwm(pwm: np.ndarray, nsites: int, bg: np.ndarray) -> np.ndarray:
    smoothed = pwm * nsites + 1.0
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    inverse = (bg ** 2) / smoothed
    return inverse / inverse.sum(axis=1, keepdims=True)


def test_spec021_make_anti_motifs_golden_formula():
    """make_anti_motifs applies smoothing and inverse-enrichment per SPEC021/SPEC024."""
    source = MemeMotif(str(TINY_MEME))
    bg = np.asarray(source.get_bg_freq(), dtype=float)
    result = make_anti_motifs(source)

    assert result.get_motif_list() == ["anti_SPEC_TINY"]
    nsites = source.get_motif_num_source_sites("SPEC_TINY")
    source_pwm = source.get_motif_pwm("SPEC_TINY")
    expected = _expected_anti_pwm(source_pwm, nsites, bg)
    actual = result.get_motif_pwm("anti_SPEC_TINY")
    assert np.allclose(actual, expected, atol=_PWM_ROW_ATOL)
    assert np.allclose(actual.sum(axis=1), 1.0, atol=_PWM_ROW_ATOL)


def test_spec021_make_anti_motifs_preserves_collection_metadata_and_order():
    """Output keeps source order, collection headers, and provenance metadata."""
    source = MemeMotif(str(SAMPLE_MEME))
    result = make_anti_motifs(source)

    assert result.get_meme_version() == source.get_meme_version()
    assert result.get_alphabet() == source.get_alphabet()
    assert result.get_strands() == source.get_strands()
    assert result.get_bg_freq() == source.get_bg_freq()
    assert result.get_motif_list() == ["anti_crp", "anti_lexA"]

    for source_name, anti_name in zip(source.get_motif_list(), result.get_motif_list()):
        assert anti_name == f"anti_{source_name}"
        assert result.get_motif_length(anti_name) == source.get_motif_length(source_name)
        assert result.get_motif_alphabet_length(anti_name) == source.get_motif_alphabet_length(
            source_name
        )
        assert result.get_motif_num_source_sites(anti_name) == source.get_motif_num_source_sites(
            source_name
        )
        assert result.get_motif_source_eval(anti_name) == source.get_motif_source_eval(source_name)


def test_spec021_make_anti_motifs_never_mutates_source():
    """Source PWM arrays and motif metadata remain unchanged after transformation."""
    source = MemeMotif(str(SAMPLE_MEME))
    motif_names = list(source.get_motif_list())
    pwm_snapshots = {name: source.get_motif_pwm(name).copy() for name in motif_names}
    bg_snapshot = list(source.get_bg_freq())
    info_snapshot = copy.deepcopy(source.motif_info_dict)

    make_anti_motifs(source)

    assert source.get_motif_list() == motif_names
    assert source.get_bg_freq() == bg_snapshot
    assert set(source.motif_info_dict) == set(info_snapshot)
    for name in motif_names:
        assert source.get_motif_num_source_sites(name) == info_snapshot[name]["num_source_sites"]
        assert source.get_motif_source_eval(name) == info_snapshot[name]["source_eval"]
        assert np.array_equal(source.get_motif_pwm(name), pwm_snapshots[name])


def test_spec021_make_anti_motifs_rejects_empty_collection():
    """Empty MEME collections fail before any motif is emitted."""
    empty = MemeMotif()
    empty.set_meme_version("4")
    empty.set_alphabet("ACGT")
    empty.set_strands(["+", "-"])
    empty.set_bg_freq([0.25, 0.25, 0.25, 0.25])

    with pytest.raises(ValueError, match="at least one motif"):
        make_anti_motifs(empty)


def test_spec021_make_anti_motifs_rejects_nonpositive_background(tmp_path):
    """Background frequencies must be finite, positive, and normalized."""
    meme = MemeMotif()
    meme.set_meme_version("4")
    meme.set_alphabet("ACGT")
    meme.set_strands(["+", "-"])
    meme.set_bg_freq([0.0, 0.5, 0.25, 0.25])
    meme.add_motif(
        "M1",
        {
            "alphabet_length": 4,
            "motif_length": 1,
            "num_source_sites": 1,
            "source_eval": 1.0,
            "pwm": np.array([[0.25, 0.25, 0.25, 0.25]]),
        },
    )

    with pytest.raises(ValueError, match="positive"):
        make_anti_motifs(meme)


def test_spec021_make_anti_motifs_rejects_invalid_pwm_rows(tmp_path):
    """Non-normalized PWM rows fail with contextual ValueError."""
    meme = MemeMotif()
    meme.set_meme_version("4")
    meme.set_alphabet("ACGT")
    meme.set_strands(["+", "-"])
    meme.set_bg_freq([0.25, 0.25, 0.25, 0.25])
    bad_pwm = np.array([[0.5, 0.1, 0.1, 0.1]])
    meme.motifs.append("BAD")
    meme.motif_info_dict["BAD"] = {
        "alphabet_length": 4,
        "motif_length": 1,
        "num_source_sites": 8,
        "source_eval": 1e-4,
        "pwm": bad_pwm.copy(),
    }

    with pytest.raises(ValueError, match="PWM rows must sum to 1.0"):
        make_anti_motifs(meme)
