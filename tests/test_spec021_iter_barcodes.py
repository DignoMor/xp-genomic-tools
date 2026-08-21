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
