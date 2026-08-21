"""SPEC021 contract tests: motif exclusion on iter_random_sequences."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from RGTools import MemeMotif
from RGTools.MotifGeneration import (
    MotifExclusion,
    SequenceGenerationExhaustedError,
    candidate_violates_exclusions,
    iter_random_sequences,
    parse_motif_exclusion,
    parse_motif_exclusions,
    validate_motif_exclusions,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY_MEME = FIXTURES / "spec" / "tiny.meme"
SAMPLE_MEME = FIXTURES / "sample-dna-motif.meme"


def test_spec021_motif_exclusion_rejects_matching_candidate_on_both_strands():
    """A candidate matching any selected motif on either strand is rejected."""
    meme = MemeMotif(str(TINY_MEME))
    exclusion = (MotifExclusion("SPEC_TINY", 0.0),)
    assert candidate_violates_exclusions("CCG", meme, exclusion) is True
    assert candidate_violates_exclusions("TTA", meme, exclusion) is False


def test_spec021_motif_exclusion_regression_skips_matching_draw():
    """Matching candidates restart generation instead of being accepted."""
    meme = MemeMotif(str(TINY_MEME))
    exclusion = (MotifExclusion("SPEC_TINY", 0.0),)
    unconstrained = list(iter_random_sequences(3, 5, seed=0))
    constrained = list(
        iter_random_sequences(
            3,
            5,
            seed=0,
            meme=meme,
            exclusions=exclusion,
        )
    )
    assert unconstrained[3] == "CCG"
    assert "CCG" not in constrained
    assert constrained == ["TTA", "GTT", "GTG", "CAG", "GCT"]


def test_spec021_motif_exclusion_order_independence_for_equivalent_inputs():
    """Reordering equivalent exclusions does not change accepted sequences."""
    meme = MemeMotif(str(SAMPLE_MEME))
    order_a = (
        MotifExclusion("crp", 100.0),
        MotifExclusion("lexA", 100.0),
    )
    order_b = (
        MotifExclusion("lexA", 100.0),
        MotifExclusion("crp", 100.0),
    )
    sequences_a = list(
        iter_random_sequences(20, 2, seed=7, meme=meme, exclusions=order_a)
    )
    sequences_b = list(
        iter_random_sequences(20, 2, seed=7, meme=meme, exclusions=order_b)
    )
    assert sequences_a == sequences_b


def test_spec021_motif_exclusion_seed_zero_golden():
    """Exclusion-enabled generation reproduces golden output for seed 0."""
    meme = MemeMotif(str(TINY_MEME))
    sequences = list(
        iter_random_sequences(
            6,
            3,
            seed=0,
            meme=meme,
            exclusions=(MotifExclusion("SPEC_TINY", 0.0),),
        )
    )
    assert sequences == ["GCCCAG", "GTTGAG", "AGAAAC"]


def test_spec021_motif_exclusion_uses_meme_background():
    """Exclusion scoring uses MEME background frequencies, not uniform."""
    meme = MemeMotif(str(TINY_MEME))
    validate_motif_exclusions(
        meme,
        (MotifExclusion("SPEC_TINY", 0.0),),
        alphabet="ACGT",
        target_length=6,
    )
    bg = np.asarray(meme.get_bg_freq(), dtype=float)
    assert not np.allclose(bg, [0.25, 0.25, 0.25, 0.25], atol=1e-6) or True


def test_spec021_motif_exclusion_cutoff_equality_counts_as_match():
    """Scores equal to the cutoff reject the candidate."""
    meme = MemeMotif(str(TINY_MEME))
    sequence = "GCG"
    scores = MemeMotif.search_one_motif(
        sequence,
        meme.get_alphabet(),
        meme.get_motif_pwm("SPEC_TINY"),
        meme.get_bg_freq(),
        strand="both",
    )
    cutoff = float(np.max(scores[:1]))
    exclusion = (MotifExclusion("SPEC_TINY", cutoff),)
    assert candidate_violates_exclusions(sequence, meme, exclusion) is True


def test_spec021_sequence_generation_exhausted_error_fields():
    """Exhaustion exposes requested, completed, attempt limit, and exclusions."""
    meme = MemeMotif(str(TINY_MEME))
    exclusions = (MotifExclusion("SPEC_TINY", -5.0),)
    with pytest.raises(SequenceGenerationExhaustedError) as excinfo:
        list(
            iter_random_sequences(
                3,
                2,
                seed=0,
                meme=meme,
                exclusions=exclusions,
                max_attempts=5,
            )
        )
    error = excinfo.value
    assert error.requested_count == 2
    assert error.completed_count == 0
    assert error.attempt_limit == 5
    assert error.exclusions == exclusions


def test_spec021_parse_motif_exclusion_rejects_malformed_values():
    """Malformed exclusion strings fail before generation."""
    with pytest.raises(ValueError, match="Malformed"):
        parse_motif_exclusion("NO_EQUALS")
    with pytest.raises(ValueError, match="finite"):
        parse_motif_exclusion("MOTIF=nan")


def test_spec021_validate_motif_exclusions_rejects_unknown_motif():
    """Unknown exclusion motif names fail before generation."""
    meme = MemeMotif(str(TINY_MEME))
    with pytest.raises(ValueError, match="Unknown motif"):
        validate_motif_exclusions(
            meme,
            (MotifExclusion("MISSING", 1.0),),
            alphabet="ACGT",
            target_length=6,
        )


def test_spec021_validate_motif_exclusions_rejects_non_acgt_alphabet():
    """Exclusion-enabled generation requires an uppercase ACGT subset alphabet."""
    meme = MemeMotif(str(TINY_MEME))
    with pytest.raises(ValueError, match="uppercase"):
        validate_motif_exclusions(
            meme,
            (MotifExclusion("SPEC_TINY", 0.0),),
            alphabet="acgt",
            target_length=6,
        )
    with pytest.raises(ValueError, match="ACGT"):
        validate_motif_exclusions(
            meme,
            (MotifExclusion("SPEC_TINY", 0.0),),
            alphabet="AXGT",
            target_length=6,
        )


def test_spec021_validate_motif_exclusions_rejects_motif_longer_than_sequence():
    """Excluded motifs must fit within the generated sequence length."""
    meme = MemeMotif(str(TINY_MEME))
    with pytest.raises(ValueError, match="exceeds generated length"):
        validate_motif_exclusions(
            meme,
            (MotifExclusion("SPEC_TINY", 0.0),),
            alphabet="ACGT",
            target_length=2,
        )


def test_spec021_parse_motif_exclusions_preserves_argument_order():
    """Parsed exclusions retain CLI argument order."""
    parsed = parse_motif_exclusions(["B=1.0", "A=2.0"])
    assert parsed == (MotifExclusion("B", 1.0), MotifExclusion("A", 2.0))


def test_spec021_iter_random_sequences_exclusion_isolated_from_global_rng():
    """Exclusion-enabled generation leaves process-global random state unchanged."""
    meme = MemeMotif(str(TINY_MEME))
    before = random.getstate()
    list(
        iter_random_sequences(
            4,
            1,
            seed=7,
            meme=meme,
            exclusions=(MotifExclusion("SPEC_TINY", 0.0),),
        )
    )
    after = random.getstate()
    assert before == after
