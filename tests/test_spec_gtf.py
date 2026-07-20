"""Spec-derived tests for RGTools GTF streaming (SPEC002, SPEC003, SPEC008)."""

from __future__ import annotations

from pathlib import Path

import pytest

from RGTools.exceptions import (
    GTFHandleFilterException,
    GTFRecordNoFeatureException,
    RGToolsInternalException,
)
from RGTools.GTF_utils import GTFHandle, GTFRecord

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spec"
TINY_GTF = FIXTURES / "tiny_gencode.gtf"


# ---------------------------------------------------------------------------
# Comments, iteration, strand
# ---------------------------------------------------------------------------


def test_gtf_handle_collects_comments_and_iterates_records():
    handle = GTFHandle(str(TINY_GTF))
    comments = handle.get_comments()
    # Leading '#' stripped once; GENCODE '##' headers keep one '#'.
    assert "description: tiny GENCODE-like fixture for SPEC008" in comments
    assert "provider GENCODE" in comments
    assert "comment only" in comments

    records = list(handle)
    assert len(records) == 4
    assert [r.search_general_info("feature_type") for r in records] == [
        "gene",
        "transcript",
        "exon",
        "gene",
    ]
    assert [r.search_general_info("strand") for r in records] == ["+", "+", "+", "-"]


def test_gtf_record_strand_must_be_plus_or_minus():
    with pytest.raises(RGToolsInternalException):
        GTFRecord(
            chr_name="chr1",
            record_source="HAVANA",
            feature_type="gene",
            start_loc=1,
            end_loc=10,
            score=".",
            strand=".",
            phase=".",
            add_info={},
        )


def test_gtf_coords_not_converted_to_bed():
    """SPEC002/SPEC008: GTF integers are stored as-read — no BED conversion.

    Fixture gene start/end are GTF 1-based closed values 11869–14409.
    If converted to BED half-open they would become 11868–14409; assert the
    stored ints match the file literally.
    """
    gene = next(
        r
        for r in GTFHandle(str(TINY_GTF))
        if r.search_general_info("feature_type") == "gene"
        and r.search_add_info("gene_name") == "DDX11L1"
    )
    assert gene.search_general_info("start_loc") == 11869
    assert gene.search_general_info("end_loc") == 14409


# ---------------------------------------------------------------------------
# Field access exceptions
# ---------------------------------------------------------------------------


def test_search_general_info_missing_raises():
    record = GTFRecord(
        chr_name="chr1",
        record_source="HAVANA",
        feature_type="gene",
        start_loc=1,
        end_loc=10,
        score=".",
        strand="+",
        phase=".",
        add_info={"gene_id": "ENSG1"},
    )
    with pytest.raises(GTFRecordNoFeatureException):
        record.search_general_info("not_a_column")


def test_search_add_info_missing_raises():
    record = next(iter(GTFHandle(str(TINY_GTF))))
    with pytest.raises(GTFRecordNoFeatureException):
        record.search_add_info("not_an_attribute")


# ---------------------------------------------------------------------------
# Filters and AND composition
# ---------------------------------------------------------------------------


def test_filter_by_general_record():
    genes = list(
        GTFHandle(str(TINY_GTF)).filter_by_general_record("feature_type", "gene")
    )
    assert len(genes) == 2
    assert {g.search_add_info("gene_name") for g in genes} == {"DDX11L1", "WASH7P"}


def test_filter_by_add_record():
    ensg1 = list(
        GTFHandle(str(TINY_GTF)).filter_by_add_record("gene_id", "ENSG00000000001")
    )
    assert [r.search_general_info("feature_type") for r in ensg1] == [
        "gene",
        "transcript",
        "exon",
    ]


def test_filter_and_composition():
    matched = list(
        GTFHandle(str(TINY_GTF))
        .filter_by_general_record("feature_type", "gene")
        .filter_by_add_record("gene_name", "WASH7P")
    )
    assert len(matched) == 1
    assert matched[0].search_general_info("strand") == "-"
    assert matched[0].search_add_info("gene_id") == "ENSG00000000002"


def test_filter_missing_attribute_raises_handle_filter_exception():
    # gene records lack transcript_id; filter probes raise wrapped exception.
    with pytest.raises(GTFHandleFilterException):
        list(
            GTFHandle(str(TINY_GTF)).filter_by_add_record(
                "transcript_id", "ENST00000000001"
            )
        )
