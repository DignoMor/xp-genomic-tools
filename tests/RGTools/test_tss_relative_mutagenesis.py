import numpy as np
import pytest

from RGTools.TSSRelativeMutagenesis import (
    MutationRegion,
    MutationRound,
    MutationTarget,
    compute_tss_relative_mutagenesis,
)


def _region(sequence="ACGTACGTAC"):
    return MutationRegion(1, "chr1", 100, 110, "r1", 105, 104, sequence)


def test_computation_joins_later_targets_by_id_and_orders_events_by_sequence():
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    regions = (_region(),)
    first = MutationRound(
        "first", "+", (1,),
        (MutationTarget("tA", "AAA"), MutationTarget("tB", "CCC")), 1,
    )
    later = MutationRound(
        "later", "+", (1,),
        (MutationTarget("tB", "GG"), MutationTarget("tA", "TT")), 2,
    )
    result = compute_tss_relative_mutagenesis(regions, (first, later), write_replaced_windows=True)
    assert [item.target_id for item in result.sequences] == ["tA", "tB"]
    assert [item.sequence for item in result.sequences] == ["ACGTATTAAC", "ACGTAGGCAC"]
    assert [(event.target_id, event.round_id) for event in result.events] == [
        ("tA", "first"), ("tA", "later"), ("tB", "first"), ("tB", "later")
    ]
    assert result.replaced_windows["later"] == (
        ("r000001|chr1:100-110|target=tA", "AA"),
        ("r000001|chr1:100-110|target=tB", "CC"),
    )


def test_computation_preserves_inputs_and_supports_strand_orientation():
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    region = _region("ACGTACGTAC")
    target = MutationTarget("t1", "ATy")
    round_spec = MutationRound("minus", "-", (-1,), (target,), 1)
    result = compute_tss_relative_mutagenesis((region,), (round_spec,), output_orientation="strand")
    assert region.sequence == "ACGTACGTAC"
    assert target.sequence == "ATy"
    assert result.sequences[0].sequence == "GTACATyCGT"


def test_placement_failure_names_first_target_for_shared_length():
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    regions = (_region(),)
    round_spec = MutationRound(
        "bad", "+", (0,),
        (MutationTarget("tA", "AAA"), MutationTarget("tB", "CCC")), 1,
    )
    try:
        compute_tss_relative_mutagenesis(regions, (round_spec,))
    except ValueError as exc:
        assert str(exc) == "TSS-relative coordinate zero is invalid (round=bad, region_row=1, target=tA, coord=0)."
    else:
        raise AssertionError("expected placement failure")


def test_multiple_regions_duplicate_intervals_have_stable_cardinality_and_ids():
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    regions = (
        _region(),
        MutationRegion(2, "chr1", 100, 110, "r2", 105, 104, "ACGTACGTAC"),
    )
    rounds = (MutationRound("r1", "+", np.array([[1], [1]], dtype=np.int64),
                           (MutationTarget("tA", "AAA"), MutationTarget("tB", "CCC")), 1),)
    result = compute_tss_relative_mutagenesis(regions, rounds)
    assert len(result.sequences) == 4
    assert [item.sequence_id for item in result.sequences] == [
        "r000001|chr1:100-110|target=tA", "r000001|chr1:100-110|target=tB",
        "r000002|chr1:100-110|target=tA", "r000002|chr1:100-110|target=tB",
    ]


@pytest.mark.parametrize("coordinates", [
    np.array([1], dtype=np.int64),
    np.array([[1, 1]], dtype=np.int64),
    np.array([[1]], dtype=np.float64),
])
def test_loaded_coordinate_shape_and_dtype_are_validated(coordinates):
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    with pytest.raises(ValueError, match="coordinate"):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (MutationRound("r1", "+", coordinates, (MutationTarget("t1", "AAA"),), 1),),
        )


def test_invalid_target_records_and_round_ids_are_rejected_before_mutation():
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    with pytest.raises(ValueError, match="Non-IUPAC"):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (MutationRound("r1", "+", (1,), (MutationTarget("t1", "A.Z"),), 1),),
        )
    with pytest.raises(ValueError, match="equal length"):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (MutationRound("r1", "+", (1,),
                           (MutationTarget("t1", "AAA"), MutationTarget("t2", "CC")), 1),),
        )
    with pytest.raises(ValueError, match="Duplicate round_id"):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 1),
             MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 2)),
        )


@pytest.mark.parametrize("field", ["fwd_tss", "rev_tss"])
def test_contextual_tss_and_bounds_failures(field):
    """SPEC027 validates loaded mutation inputs and ordered trajectory results."""
    kwargs = {"fwd_tss": 105, "rev_tss": 104}
    kwargs[field] = -1
    region = MutationRegion(1, "chr1", 100, 110, "r1", **kwargs, sequence="ACGTACGTAC")
    with pytest.raises(ValueError, match="missing"):
        compute_tss_relative_mutagenesis(
            (region,), (MutationRound("r1", "+" if field == "fwd_tss" else "-", (1,),
                                      (MutationTarget("t1", "AAA"),), 1),)
        )


def test_SPEC027_empty_regions_return_empty_complete_result():
    """SPEC027 permits valid empty region collections and preserves round artifacts."""
    coordinates = np.empty((0, 1), dtype=np.int64)
    round_spec = MutationRound(
        "empty", "+", coordinates, (MutationTarget("t1", "AAA"),), 1
    )
    result = compute_tss_relative_mutagenesis(
        (), (round_spec,), write_replaced_windows=True
    )
    assert result.sequences == ()
    assert result.events == ()
    assert result.replaced_windows == {"empty": ()}


@pytest.mark.parametrize(
    "rounds, message",
    [
        ((), "at least one round"),
        (
            (MutationRound("empty", "+", (1,), (), 1),),
            "target FASTA is empty",
        ),
    ],
)
def test_SPEC027_empty_regions_still_validate_invalid_rounds(rounds, message):
    """SPEC027 validates rounds even when no regions reach placement."""
    with pytest.raises(ValueError, match=message):
        compute_tss_relative_mutagenesis((), rounds)


def test_SPEC027_does_not_mutate_coordinate_arrays_or_loaded_records():
    """SPEC027 leaves caller-owned coordinates, regions, and targets unchanged."""
    coordinates = np.array([[1]], dtype=np.int64)
    original_coordinates = coordinates.copy()
    region = _region()
    target = MutationTarget("t1", "ATy")
    round_spec = MutationRound("minus", "-", coordinates, (target,), 1)
    compute_tss_relative_mutagenesis((region,), (round_spec,))
    assert np.array_equal(coordinates, original_coordinates)
    assert region.sequence == "ACGTACGTAC"
    assert target.sequence == "ATy"


def test_SPEC027_default_and_explicit_genomic_orientation_are_equal():
    """SPEC027 default genomic output equals explicit genomic orientation."""
    rounds = (
        MutationRound("plus", "+", (1,), (MutationTarget("t1", "AAA"),), 1),
    )
    implicit = compute_tss_relative_mutagenesis((_region(),), rounds)
    explicit = compute_tss_relative_mutagenesis(
        (_region(),), rounds, output_orientation="genomic"
    )
    assert implicit.sequences == explicit.sequences
    assert implicit.events == explicit.events


def test_SPEC027_minus_orientation_preserves_case_and_genomic_audit_windows():
    """SPEC027 keeps minus audit windows genomic-forward while orienting finals."""
    rounds = (
        MutationRound("first", "-", (-1,), (MutationTarget("t1", "ATy"),), 1),
        MutationRound("second", "-", (-1,), (MutationTarget("t1", "TT"),), 2),
    )
    genomic = compute_tss_relative_mutagenesis(
        (_region(),), rounds, write_replaced_windows=True
    )
    strand = compute_tss_relative_mutagenesis(
        (_region(),), rounds, output_orientation="strand", write_replaced_windows=True
    )
    assert genomic.sequences[0].sequence == "ACGrAAGTAC"
    assert strand.sequences[0].sequence == "GTACTTyCGT"
    assert genomic.replaced_windows == strand.replaced_windows == {
        "first": (("r000001|chr1:100-110|target=t1", "TAC"),),
        "second": (("r000001|chr1:100-110|target=t1", "AT"),),
    }
    assert genomic.events == strand.events


@pytest.mark.parametrize(
    "target, message",
    [
        (MutationTarget("", "AAA"), "must be nonempty"),
        (MutationTarget("bad id", "AAA"), "whitespace"),
        (MutationTarget("bad|id", "AAA"), "delimiter"),
        (MutationTarget("t1", ""), "Empty mutation target"),
    ],
)
def test_SPEC027_rejects_empty_or_unsafe_target_ids_and_sequences(target, message):
    """SPEC027 rejects malformed target records before mutation."""
    with pytest.raises(ValueError, match=message):
        compute_tss_relative_mutagenesis(
            (_region(),), (MutationRound("r1", "+", (1,), (target,), 1),)
        )


def test_SPEC027_rejects_duplicate_target_ids_and_later_set_mismatch():
    """SPEC027 requires unique target IDs and identical target sets across rounds."""
    with pytest.raises(ValueError, match="duplicate target ID"):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (
                MutationRound(
                    "r1", "+", (1,),
                    (MutationTarget("t1", "AAA"), MutationTarget("t1", "CCC")), 1,
                ),
            ),
        )
    with pytest.raises(ValueError, match="must match first-round set"):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (
                MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 1),
                MutationRound("r2", "+", (1,), (MutationTarget("t2", "AAA"),), 2),
            ),
        )


def test_SPEC027_rejects_coordinate_cardinality_mismatch():
    """SPEC027 requires one coordinate per input region in every round."""
    regions = (
        _region(),
        MutationRegion(2, "chr1", 100, 110, "r2", 105, 104, "ACGTACGTAC"),
    )
    with pytest.raises(ValueError, match=r"shape \(2, 1\)"):
        compute_tss_relative_mutagenesis(
            regions,
            (MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 1),),
        )


@pytest.mark.parametrize(
    "region, target, message",
    [
        (
            MutationRegion(1, "chr1", 100, 110, "r1", 99, 104, "ACGTACGTAC"),
            "AAA",
            r"fwdTSS=99.*round=r1, region_row=1, target=t1, coord=1",
        ),
        (
            MutationRegion(1, "chr1", 100, 110, "r1", 105, 104, "ACGTACGTAC"),
            "A" * 11,
            r"Replacement window does not fit.*round=r1, region_row=1, target=t1, coord=1",
        ),
    ],
)
def test_SPEC027_outside_tss_and_replacement_bounds_are_contextual(region, target, message):
    """SPEC027 reports exact round, row, target, and coordinate placement context."""
    with pytest.raises(ValueError, match=message):
        compute_tss_relative_mutagenesis(
            (region,), (MutationRound("r1", "+", (1,), (MutationTarget("t1", target),), 1),)
        )


def test_SPEC027_placement_failure_uses_first_target_after_later_reordering():
    """SPEC027 keeps first-target diagnostics when a later round is invalid."""
    with pytest.raises(
        ValueError,
        match=r"round=later, region_row=1, target=tA, coord=0",
    ):
        compute_tss_relative_mutagenesis(
            (_region(),),
            (
                MutationRound(
                    "first", "+", (1,),
                    (MutationTarget("tA", "AAA"), MutationTarget("tB", "CCC")), 1,
                ),
                MutationRound(
                    "later", "+", (0,),
                    (MutationTarget("tB", "GG"), MutationTarget("tA", "TT")), 2,
                ),
            ),
        )


def test_SPEC027_rejects_noncontiguous_region_and_round_ordinals():
    """SPEC027 requires paired region rows and ordered round indexes."""
    with pytest.raises(ValueError, match="Region rows"):
        compute_tss_relative_mutagenesis(
            (MutationRegion(2, "chr1", 100, 110, "r1", 105, 104, "ACGTACGTAC"),),
            (MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 1),),
        )
    with pytest.raises(ValueError, match="Round indexes"):
        compute_tss_relative_mutagenesis(
            (_region(),), (MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 2),)
        )


def test_SPEC027_rejects_short_paired_region_sequence():
    """SPEC027 validates that each paired sequence spans its region interval."""
    with pytest.raises(ValueError, match="does not match interval length"):
        compute_tss_relative_mutagenesis(
            (MutationRegion(1, "chr1", 100, 110, "r1", 105, 104, "ACGTACGTA"),),
            (MutationRound("r1", "+", (1,), (MutationTarget("t1", "AAA"),), 1),),
        )


def test_SPEC027_rejects_mixed_strands_for_strand_output():
    """SPEC027 requires one unique round strand for strand-oriented output."""
    rounds = (
        MutationRound("plus", "+", (1,), (MutationTarget("t1", "AAA"),), 1),
        MutationRound("minus", "-", (1,), (MutationTarget("t1", "AAA"),), 2),
    )
    with pytest.raises(ValueError, match="requires every round to declare the same strand"):
        compute_tss_relative_mutagenesis((_region(),), rounds, output_orientation="strand")
