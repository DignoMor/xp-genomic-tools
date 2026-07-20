"""SPEC002 / SPEC006 black-box tests for MemeMotif (MEME subset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from RGTools import MemeMotif

FIXTURES_SPEC = Path(__file__).resolve().parent / "fixtures" / "spec"
TINY_MEME = FIXTURES_SPEC / "tiny.meme"

# Expected PWM for SPEC_TINY in the vendored fixture (rows sum to 1).
EXPECTED_PWM = np.array(
    [
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ],
    dtype=float,
)


def test_parse_vendored_meme_subset():
    mm = MemeMotif(str(TINY_MEME))
    assert mm.get_meme_version() == "4"
    assert mm.get_alphabet() == "ACGT"
    assert mm.get_strands() == ["+", "-"]
    assert mm.get_bg_freq() == [0.25, 0.25, 0.25, 0.25]
    assert mm.get_motif_list() == ["SPEC_TINY"]
    assert mm.get_motif_length("SPEC_TINY") == 3
    assert mm.get_motif_alphabet_length("SPEC_TINY") == 4
    assert mm.get_motif_num_source_sites("SPEC_TINY") == 8
    np.testing.assert_allclose(mm.get_motif_pwm("SPEC_TINY"), EXPECTED_PWM, atol=1e-6)


def test_write_read_round_trip(tmp_path):
    mm = MemeMotif(str(TINY_MEME))
    out = tmp_path / "roundtrip.meme"
    mm.write_meme_file(str(out))

    reloaded = MemeMotif(str(out))
    assert reloaded.get_meme_version() == mm.get_meme_version()
    assert reloaded.get_alphabet() == mm.get_alphabet()
    assert reloaded.get_strands() == mm.get_strands()
    assert reloaded.get_bg_freq() == mm.get_bg_freq()
    assert reloaded.get_motif_list() == mm.get_motif_list()
    np.testing.assert_allclose(
        reloaded.get_motif_pwm("SPEC_TINY"),
        mm.get_motif_pwm("SPEC_TINY"),
        atol=1e-6,
    )
    assert reloaded.get_motif_num_source_sites("SPEC_TINY") == mm.get_motif_num_source_sites(
        "SPEC_TINY"
    )
    assert reloaded.get_motif_source_eval("SPEC_TINY") == mm.get_motif_source_eval("SPEC_TINY")


def test_add_motif_rejects_unnormalized_pwm_rows():
    mm = MemeMotif()
    mm.set_meme_version("4")
    mm.set_alphabet("ACGT")
    mm.set_strands(["+", "-"])
    mm.set_bg_freq([0.25, 0.25, 0.25, 0.25])

    bad_pwm = EXPECTED_PWM.copy()
    bad_pwm[0, 0] = 0.9  # row sum ≠ 1
    motif_info = {
        "alphabet_length": 4,
        "motif_length": 3,
        "num_source_sites": 4,
        "source_eval": 0.01,
        "pwm": bad_pwm,
    }
    with pytest.raises(ValueError):
        mm.add_motif("BAD_ROW", motif_info)


def test_add_motif_accepts_normalized_pwm():
    mm = MemeMotif()
    mm.set_meme_version("4")
    mm.set_alphabet("ACGT")
    mm.set_strands(["+", "-"])
    mm.set_bg_freq([0.25, 0.25, 0.25, 0.25])

    motif_info = {
        "alphabet_length": 4,
        "motif_length": 3,
        "num_source_sites": 4,
        "source_eval": 0.01,
        "pwm": EXPECTED_PWM.copy(),
    }
    mm.add_motif("OK", motif_info)
    assert mm.get_motif_list() == ["OK"]
    np.testing.assert_allclose(mm.get_motif_pwm("OK"), EXPECTED_PWM, atol=1e-6)


def test_calculate_pwm_score_length_mismatch():
    with pytest.raises(ValueError):
        MemeMotif.calculate_pwm_score("AC", EXPECTED_PWM)


def test_calculate_pwm_score_matching_length():
    score = MemeMotif.calculate_pwm_score("ACG", EXPECTED_PWM)
    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)


def test_search_one_motif_strand_plus_minus_both():
    seq = "TTACGTTT"
    alphabet = "ACGT"
    scores_plus = MemeMotif.search_one_motif(seq, alphabet, EXPECTED_PWM, strand="+")
    scores_minus = MemeMotif.search_one_motif(seq, alphabet, EXPECTED_PWM, strand="-")
    scores_both = MemeMotif.search_one_motif(seq, alphabet, EXPECTED_PWM, strand="both")

    assert len(scores_plus) == len(seq)
    assert len(scores_minus) == len(seq)
    assert len(scores_both) == len(seq)

    # Forward and reverse strands are not identical for this sequence/PWM.
    assert not np.allclose(scores_plus, scores_minus)

    # Spec: `both` takes the max of forward and RC over scorable windows.
    motif_len = EXPECTED_PWM.shape[0]
    n_windows = len(seq) - motif_len + 1
    np.testing.assert_allclose(
        scores_both[:n_windows],
        np.maximum(scores_plus[:n_windows], scores_minus[:n_windows]),
        atol=1e-9,
    )


def test_search_one_motif_invalid_strand():
    with pytest.raises(ValueError):
        MemeMotif.search_one_motif("ACGTTT", "ACGT", EXPECTED_PWM, strand="x")
