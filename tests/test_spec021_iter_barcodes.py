"""SPEC021 contract tests: iter_barcodes enumeration."""

from __future__ import annotations

from pathlib import Path

import pytest

from RGTools import MemeMotif
from RGTools.MotifGeneration import MotifExclusion, iter_barcodes

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY_MEME = FIXTURES / "spec" / "tiny.meme"


def test_spec021_iter_barcodes_preserves_alphabet_order():
    """Barcodes enumerate the Cartesian product in supplied-alphabet order."""
    assert list(iter_barcodes(2, alphabet="XY")) == ["XX", "XY", "YX", "YY"]


def test_spec021_iter_barcodes_exact_small_cardinality():
    """Length-1 DNA barcodes enumerate exactly four candidates."""
    assert list(iter_barcodes(1, alphabet="ACGT")) == ["A", "C", "G", "T"]


def test_spec021_iter_barcodes_excludes_earlier_ordered_rc_partners():
    """RC filtering keeps the first accepted representative and palindromes."""
    assert list(iter_barcodes(2, exclude_rc_collisions=True)) == [
        "AA", "AC", "AG", "AT", "CA", "CC", "CG", "GA", "GC", "TA"
    ]
    assert list(iter_barcodes(1, exclude_rc_collisions=True)) == ["A", "C"]


def test_spec021_iter_barcodes_rc_uses_supplied_alphabet_order():
    """Partner precedence follows user order, even when lexical order differs."""
    assert list(iter_barcodes(1, alphabet="TGCA", exclude_rc_collisions=True)) == ["T", "G"]


@pytest.mark.parametrize("alphabet", ["", "a", "Ac", "N", "RY", "X", "!", "AAC"])
def test_spec021_iter_barcodes_rc_requires_unique_uppercase_dna_subset(alphabet):
    with pytest.raises(ValueError, match="alphabet"):
        list(iter_barcodes(1, alphabet=alphabet, exclude_rc_collisions=True))


def test_spec021_iter_barcodes_rc_subset_without_complement_partner():
    """Unrepresentable reverse complements do not suppress candidates."""
    assert list(iter_barcodes(2, alphabet="AC", exclude_rc_collisions=True)) == [
        "AA", "AC", "CA", "CC"
    ]


def test_spec021_iter_barcodes_rc_requires_string_alphabet():
    """SPEC021: reverse-complement mode requires a string alphabet."""
    with pytest.raises(ValueError, match="string"):
        list(iter_barcodes(1, alphabet=["A", "C"], exclude_rc_collisions=True))


def test_spec021_iter_barcodes_rc_and_motif_exclusions_are_both_applied():
    """Motif-rejected sequences do not reserve reverse-complement partners."""
    meme = MemeMotif(str(TINY_MEME))
    exclusions = (MotifExclusion("SPEC_TINY", 0.0),)
    motif_survivors = list(iter_barcodes(
        3,
        meme=meme,
        exclusions=exclusions,
    ))
    expected = []
    accepted = set()
    for candidate in motif_survivors:
        rc = "".join({"A": "T", "C": "G", "G": "C", "T": "A"}[base]
                     for base in reversed(candidate))
        if rc not in accepted:
            expected.append(candidate)
            accepted.add(candidate)
    assert list(iter_barcodes(
        3,
        meme=meme,
        exclusions=exclusions,
        exclude_rc_collisions=True,
    )) == expected


def test_spec021_iter_barcodes_rc_streams_incrementally():
    """The public iterator can yield its first result before consuming all output."""
    barcodes = iter_barcodes(8, exclude_rc_collisions=True)
    assert next(barcodes) == "AAAAAAAA"
    assert next(barcodes) == "AAAAAAAC"


def test_spec021_iter_barcodes_accepts_unused_meme_without_exclusions():
    """SPEC021: optional MEME input remains accepted when no motif filters run."""
    meme = MemeMotif(str(TINY_MEME))
    assert list(iter_barcodes(1, alphabet="AC", meme=meme,
                              exclude_rc_collisions=True)) == ["A", "C"]


def test_spec021_iter_barcodes_preflight_guard_without_large_enumeration():
    """Oversized candidate spaces fail before enumeration begins."""
    with pytest.raises(ValueError, match="candidate space size"):
        list(iter_barcodes(10, alphabet="ACGT", max_candidates=1_000_000))


def test_spec021_iter_barcodes_allows_explicit_limit_increase():
    """Raising max_candidates is the explicit opt-in to larger searches."""
    barcodes = list(iter_barcodes(5, alphabet="XY", max_candidates=64))
    assert len(barcodes) == 32


def test_spec021_iter_barcodes_applies_shared_exclusions():
    """Motif exclusions reuse random_seq semantics and Cartesian order."""
    meme = MemeMotif(str(TINY_MEME))
    barcodes = list(
        iter_barcodes(
            3,
            meme=meme,
            exclusions=(MotifExclusion("SPEC_TINY", 0.0),),
        )
    )
    assert len(barcodes) == 44
    assert barcodes[0] == "AAA"
    assert barcodes[-1] == "TTT"


def test_spec021_iter_barcodes_empty_survivors_yields_nothing():
    """An exhaustive search with no survivors yields an empty iterator."""
    meme = MemeMotif(str(TINY_MEME))
    assert (
        list(
            iter_barcodes(
                3,
                meme=meme,
                exclusions=(MotifExclusion("SPEC_TINY", -5.0),),
            )
        )
        == []
    )


def test_spec021_iter_barcodes_rejects_duplicate_alphabet():
    """Duplicate alphabet characters fail before enumeration."""
    with pytest.raises(ValueError, match="unique"):
        list(iter_barcodes(1, alphabet="AAC"))


def test_spec021_iter_barcodes_rejects_nonpositive_length():
    """Non-positive barcode lengths fail before enumeration."""
    with pytest.raises(ValueError, match="barcode_length"):
        list(iter_barcodes(0))
