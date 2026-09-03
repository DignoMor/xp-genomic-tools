"""Spec-derived tests for RGTools Elements collections (SPEC002, SPEC005)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from RGTools import ExogenousSequences, GenomicElements

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spec"
TINY_FA = FIXTURES / "tiny.fa"
REGIONS_BED3 = FIXTURES / "regions.bed3"
REGIONS_BEDGRAPH = FIXTURES / "regions.bedGraph"
REGIONS_BED3GENE = FIXTURES / "regions.bed3gene"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_genomic_elements_construct_bed3():
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        assert ge.get_num_regions() == 2
        assert ge.region_file_type == "bed3"
        assert ge.fasta_path == str(TINY_FA)
        assert ge.region_file_path == str(REGIONS_BED3)
    finally:
        ge.close()


def test_genomic_elements_invalid_region_file_type():
    with pytest.raises(ValueError, match="Invalid region file type"):
        GenomicElements(str(REGIONS_BED3), "not_a_type", str(TINY_FA))


# ---------------------------------------------------------------------------
# Order preservation (enable_sort=False) and filter annotation alignment
# ---------------------------------------------------------------------------


def test_enable_sort_false_preserves_region_order():
    """GenomicElements loads with sorting disabled so annotation row i aligns to region i."""
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        bt = ge.get_region_bed_table()
        # Fixture order is chrB then chrA (not chrom-sorted).
        assert bt.get_chrom_names().tolist() == ["chrB", "chrA"]
        assert bt.get_start_locs().tolist() == [1, 0]
        assert bt.get_end_locs().tolist() == [5, 4]
        assert ge.get_all_region_seqs() == ["GGGC", "ACGT"]
    finally:
        ge.close()


def test_filter_keeps_annotation_alignment(tmp_path):
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        # Labels tied to fixture order: index 0 = chrB, index 1 = chrA.
        ge.load_region_stat_from_arr("score", np.array([10.0, 20.0]))
        ge.load_mask_from_arr("keep_flag", np.array([True, False]))

        out_bed = tmp_path / "filtered.bed3"
        # Keep only index 0 (chrB / score 10).
        filtered = ge.apply_logical_filter(np.array([True, False]), str(out_bed))
        try:
            bt = filtered.get_region_bed_table()
            assert bt.get_chrom_names().tolist() == ["chrB"]
            assert filtered.get_stat_arr("score").ravel().tolist() == [10.0]
            assert filtered.get_mask_arr("keep_flag").ravel().tolist() == [True]
            assert filtered.get_num_regions() == 1
        finally:
            filtered.close()
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# ExogenousSequences
# ---------------------------------------------------------------------------


def test_exogenous_sequences_synthetic_bed3_view():
    es = ExogenousSequences(str(TINY_FA))
    try:
        assert es.region_file_type == "bed3"
        assert es.get_sequence_ids().tolist() == ["chrA", "chrB"]

        bt = es.get_region_bed_table()
        assert bt.get_chrom_names().tolist() == ["chrA", "chrB"]
        assert bt.get_start_locs().tolist() == [0, 0]
        assert bt.get_end_locs().tolist() == [8, 8]
        assert es.get_all_region_seqs() == ["ACGTACGT", "GGGGCCCC"]
    finally:
        es.close()


def test_exogenous_sequences_region_file_path_not_implemented():
    es = ExogenousSequences(str(TINY_FA))
    try:
        with pytest.raises(NotImplementedError):
            _ = es.region_file_path
    finally:
        es.close()


# ---------------------------------------------------------------------------
# Annotation types, shapes, NPZ single-array, type not in file
# ---------------------------------------------------------------------------


def test_annotation_stat_mask_array_shape_rules(tmp_path):
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        # stat: (N,) or (N, 1) → stored (N, 1)
        ge.load_region_stat_from_arr("s1", np.array([1.0, 2.0]))
        assert ge.get_anno_type("s1") == "stat"
        assert ge.get_stat_arr("s1").shape == (2, 1)

        ge.load_region_stat_from_arr("s2", np.array([[3.0], [4.0]]))
        assert ge.get_stat_arr("s2").shape == (2, 1)

        with pytest.raises(ValueError, match="Stat array"):
            ge.load_region_stat_from_arr("s_bad", np.zeros((2, 2)))

        # mask: bool dtype; (N,) or (N, 1) → (N, 1)
        ge.load_mask_from_arr("m1", np.array([True, False]))
        assert ge.get_anno_type("m1") == "mask"
        assert ge.get_mask_arr("m1").shape == (2, 1)
        assert ge.get_mask_arr("m1").dtype == bool

        with pytest.raises(ValueError, match="boolean"):
            ge.load_mask_from_arr("m_bad", np.array([1, 0]))

        # array: (N, …) with ≥2 dims
        ge.load_region_array_from_arr("a1", np.zeros((2, 3, 4)))
        assert ge.get_anno_type("a1") == "array"
        assert ge.get_arr_anno("a1").shape == (2, 3, 4)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            ge.load_region_array_from_arr("a_bad", np.zeros(2))

        with pytest.raises(ValueError, match="does not match number of regions"):
            ge.load_region_array_from_arr("a_n", np.zeros((3, 2, 2)))

        # track: per-region length must match region length
        ge.load_region_track_from_list(
            "t1",
            [np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0])],
        )
        assert ge.get_anno_type("t1") == "track"
        assert [len(x) for x in ge.get_track_list("t1")] == [4, 4]

        with pytest.raises(ValueError, match="Annotation length"):
            ge.load_region_track_from_list(
                "t_bad",
                [np.ones(3), np.ones(4)],
            )
    finally:
        ge.close()


def test_annotation_type_not_stored_in_npy(tmp_path):
    """Caller supplies anno_type on load; type is not embedded in the npy/npz file."""
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        ge.load_region_stat_from_arr("score", np.array([1.0, 2.0]))
        npy_path = tmp_path / "score.npy"
        ge.save_anno_npy("score", str(npy_path))
    finally:
        ge.close()

    ge2 = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        ge2.load_region_anno_from_npy("as_stat", str(npy_path), anno_type="stat")
        assert ge2.get_anno_type("as_stat") == "stat"
        np.testing.assert_array_equal(ge2.get_stat_arr("as_stat").ravel(), [1.0, 2.0])

        # Same bytes loaded as mask fail because dtype is not bool (type comes from caller).
        with pytest.raises(ValueError, match="boolean"):
            ge2.load_region_anno_from_npy("as_mask", str(npy_path), anno_type="mask")
    finally:
        ge2.close()


def test_npz_must_contain_exactly_one_array(tmp_path):
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        multi = tmp_path / "multi.npz"
        np.savez(multi, a=np.array([1.0, 2.0]), b=np.array([3.0, 4.0]))
        with pytest.raises(ValueError, match="multiple arrays"):
            ge.load_region_anno_from_npy("x", str(multi), anno_type="stat")

        single = tmp_path / "single.npz"
        np.savez(single, only=np.array([10.0, 20.0]))
        ge.load_region_anno_from_npy("from_npz", str(single), anno_type="stat")
        np.testing.assert_array_equal(ge.get_stat_arr("from_npz").ravel(), [10.0, 20.0])

        # Round-trip save writes a single-array NPZ.
        out_npz = tmp_path / "saved.npz"
        ge.save_anno_npz("from_npz", str(out_npz))
        with np.load(out_npz) as z:
            assert len(z.files) == 1
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------


def test_one_hot_alphabet_and_ambiguous():
    oh = GenomicElements.one_hot_encoding("ACGT")
    assert oh.shape == (4, 4)
    np.testing.assert_array_equal(oh, np.eye(4, dtype=oh.dtype))

    # Ambiguous IUPAC → all-zero rows.
    for base in ("N", "R", "Y", "W", "S", "M", "K"):
        row = GenomicElements.one_hot_encoding(base)
        assert row.shape == (1, 4)
        assert np.all(row == 0)

    # Non-ambiguous illegal base → AssertionError (SPEC005).
    with pytest.raises(AssertionError):
        GenomicElements.one_hot_encoding("Z")


def test_bulk_one_hot_requires_length_homogeneous():
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        # Both regions length 4.
        oh = ge.get_all_region_one_hot()
        assert oh.shape == (2, 4, 4)
        np.testing.assert_array_equal(oh[0], GenomicElements.one_hot_encoding("GGGC"))
        np.testing.assert_array_equal(oh[1], GenomicElements.one_hot_encoding("ACGT"))
    finally:
        ge.close()


def test_bulk_one_hot_rejects_heterogeneous_lengths(tmp_path):
    bed = tmp_path / "hetero.bed3"
    bed.write_text("chrA\t0\t4\nchrA\t0\t6\n")
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match="length-homogeneous"):
            ge.get_all_region_one_hot()
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# export_exogenous_sequences
# ---------------------------------------------------------------------------


def test_export_exogenous_sequences_header_and_refuse_existing(tmp_path):
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        out = tmp_path / "exported.fa"
        ge.export_exogenous_sequences(str(out))
        text = out.read_text()
        assert text == ">chrB:1-5\nGGGC\n>chrA:0-4\nACGT\n"

        before = out.read_bytes()
        with pytest.raises(ValueError, match="already exists"):
            ge.export_exogenous_sequences(str(out))
        assert out.read_bytes() == before
    finally:
        ge.close()


def test_export_exogenous_sequences_rejects_negative_start(tmp_path):
    """start < 0 raises contextual ValueError; destination stays absent (SPEC005)."""
    bed = tmp_path / "neg.bed3"
    bed.write_text("chrA\t-1\t4\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match=r"chrA:-1-4"):
            ge.export_exogenous_sequences(str(out))
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_sequences_rejects_zero_width(tmp_path):
    """start >= end raises contextual ValueError; destination stays absent (SPEC005)."""
    bed = tmp_path / "zero.bed3"
    bed.write_text("chrA\t4\t4\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match=r"chrA:4-4"):
            ge.export_exogenous_sequences(str(out))
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_sequences_rejects_end_past_chromosome(tmp_path):
    """end > chromosome length fails instead of truncating (SPEC005)."""
    bed = tmp_path / "past.bed3"
    # tiny.fa chrA length is 8; end=9 would silently truncate under slicing.
    bed.write_text("chrA\t0\t9\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match=r"chrA:0-9"):
            ge.export_exogenous_sequences(str(out))
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_sequences_rejects_missing_chrom_after_valid_row(tmp_path):
    """Missing chrom after a valid row leaves destination absent (SPEC005)."""
    bed = tmp_path / "mixed.bed3"
    bed.write_text("chrA\t0\t4\nchrMissing\t0\t4\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match=r"chrMissing"):
            ge.export_exogenous_sequences(str(out))
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_sequences_allows_exact_chromosome_end(tmp_path):
    """end == chromosome length remains valid and exports the full slice (SPEC005)."""
    bed = tmp_path / "exact.bed3"
    bed.write_text("chrA\t0\t8\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        ge.export_exogenous_sequences(str(out))
        assert out.read_text() == ">chrA:0-8\nACGTACGT\n"
    finally:
        ge.close()


def test_export_exogenous_sequences_validates_bedgraph_schema(tmp_path):
    """Registered schemas other than bed3 receive the same containment checks (SPEC005)."""
    bed = tmp_path / "past.bedGraph"
    bed.write_text("chrA\t0\t9\t1.0\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bedGraph", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match=r"chrA:0-9"):
            ge.export_exogenous_sequences(str(out))
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_default_matches_explicit_genomic_coordinate(tmp_path):
    """Omitting modes matches explicit genomic+coordinate (SPEC005 / #13)."""
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        default_out = tmp_path / "default.fa"
        explicit_out = tmp_path / "explicit.fa"
        ge.export_exogenous_sequences(str(default_out))
        ge.export_exogenous_sequences(
            str(explicit_out),
            output_orientation="genomic",
            record_id="coordinate",
        )
        assert default_out.read_bytes() == explicit_out.read_bytes()
        assert default_out.read_text() == ">chrB:1-5\nGGGC\n>chrA:0-4\nACGT\n"
    finally:
        ge.close()


def test_export_exogenous_strand_orientation_plus_and_minus(tmp_path):
    """Strand mode keeps + sequences and reverse-complements - (SPEC005 / #13)."""
    bed6 = tmp_path / "mixed.bed6"
    bed6.write_text(
        "chrA\t0\t4\tr_plus\t1\t+\n"
        "chrB\t0\t4\tr_minus\t1\t-\n"
    )
    ge = GenomicElements(str(bed6), "bed6", str(TINY_FA))
    try:
        out = tmp_path / "strand.fa"
        ge.export_exogenous_sequences(str(out), output_orientation="strand")
        assert out.read_text() == (
            ">chrA:0-4\nACGT\n"
            ">chrB:0-4\nCCCC\n"
        )
    finally:
        ge.close()


def test_export_exogenous_genomic_ignores_invalid_strand(tmp_path):
    """Genomic orientation ignores unused strand values including '.' (SPEC005 / #13)."""
    bed6 = tmp_path / "dot.bed6"
    bed6.write_text("chrA\t0\t4\tr_dot\t1\t.\n")
    ge = GenomicElements(str(bed6), "bed6", str(TINY_FA))
    try:
        out = tmp_path / "genomic.fa"
        ge.export_exogenous_sequences(str(out), output_orientation="genomic")
        assert out.read_text() == ">chrA:0-4\nACGT\n"
    finally:
        ge.close()


def test_export_exogenous_strand_rejects_dot_strand(tmp_path):
    """Strand orientation rejects '.' / missing strand before publish (SPEC005 / #13)."""
    bed6 = tmp_path / "dot.bed6"
    bed6.write_text("chrA\t0\t4\tr_dot\t1\t.\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed6), "bed6", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match="strand"):
            ge.export_exogenous_sequences(str(out), output_orientation="strand")
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_strand_rejects_strandless_schema(tmp_path):
    """Strand orientation rejects schemas without row-level strand (SPEC005 / #13)."""
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match="strand"):
            ge.export_exogenous_sequences(str(out), output_orientation="strand")
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_record_id_name_preserves_order(tmp_path):
    """Name identity preserves names and row order (SPEC005 / #13)."""
    bed6 = tmp_path / "named.bed6"
    bed6.write_text(
        "chrA\t0\t4\tpeakA\t1\t+\n"
        "chrB\t0\t4\tpeakB\t1\t-\n"
    )
    ge = GenomicElements(str(bed6), "bed6", str(TINY_FA))
    try:
        out = tmp_path / "named.fa"
        ge.export_exogenous_sequences(
            str(out),
            output_orientation="strand",
            record_id="name",
        )
        assert out.read_text() == (
            ">peakA\nACGT\n"
            ">peakB\nCCCC\n"
        )
    finally:
        ge.close()


def test_export_exogenous_name_without_strand_ok(tmp_path):
    """Name identity works on TREbed without requiring strand (SPEC005 / #13)."""
    tre = tmp_path / "regions.trebed"
    tre.write_text("chrA\t0\t4\tentityA\t0\t-1\n")
    ge = GenomicElements(str(tre), "TREbed", str(TINY_FA))
    try:
        out = tmp_path / "named.fa"
        ge.export_exogenous_sequences(str(out), record_id="name")
        assert out.read_text() == ">entityA\nACGT\n"
    finally:
        ge.close()


def test_export_exogenous_rejects_duplicate_coordinate_ids(tmp_path):
    """Coordinate identity rejects duplicate chrom:start-end IDs (SPEC005 / #13)."""
    bed = tmp_path / "dup.bed3"
    bed.write_text("chrA\t0\t4\nchrA\t0\t4\n")
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match="Duplicate"):
            ge.export_exogenous_sequences(str(out))
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_rejects_duplicate_names(tmp_path):
    """Name identity rejects exact duplicate names (SPEC005 / #13)."""
    bed6 = tmp_path / "dup.bed6"
    bed6.write_text(
        "chrA\t0\t4\tsame\t1\t+\n"
        "chrB\t0\t4\tsame\t1\t-\n"
    )
    out = tmp_path / "out.fa"
    ge = GenomicElements(str(bed6), "bed6", str(TINY_FA))
    try:
        with pytest.raises(ValueError, match="Duplicate"):
            ge.export_exogenous_sequences(str(out), record_id="name")
        assert not out.exists()
    finally:
        ge.close()


def test_export_exogenous_strand_rejects_non_iupac(tmp_path):
    """Strand mode rejects non-IUPAC symbols on + and - rows (SPEC005 / #13)."""
    fa = tmp_path / "bad.fa"
    fa.write_text(">chrZ\nACGTZACG\n")
    out = tmp_path / "out.fa"

    for strand in ("+", "-"):
        bed6 = tmp_path / f"{strand}.bed6"
        bed6.write_text(f"chrZ\t0\t8\tr1\t1\t{strand}\n")
        ge = GenomicElements(str(bed6), "bed6", str(fa))
        try:
            with pytest.raises(ValueError, match="[Nn]on-IUPAC|non-IUPAC"):
                ge.export_exogenous_sequences(str(out), output_orientation="strand")
            assert not out.exists()
        finally:
            ge.close()


def test_export_exogenous_iupac_case_preserving_reverse_complement(tmp_path):
    """Minus-strand reverse complement preserves case for IUPAC bases (SPEC005 / #13)."""
    fa = tmp_path / "iupac.fa"
    fa.write_text(">chrZ\nACgt\n")
    bed6 = tmp_path / "minus.bed6"
    bed6.write_text("chrZ\t0\t4\tr1\t1\t-\n")
    ge = GenomicElements(str(bed6), "bed6", str(fa))
    try:
        out = tmp_path / "out.fa"
        ge.export_exogenous_sequences(str(out), output_orientation="strand")
        assert out.read_text() == ">chrZ:0-4\nacGT\n"
    finally:
        ge.close()


def test_export_exogenous_rejects_invalid_names(tmp_path):
    """Name identity rejects '.', whitespace, and similar invalid names (SPEC005 / #13)."""
    cases = [
        ("chrA\t0\t4\t.\t1\t+\n", "name"),
        ("chrA\t0\t4\tbad name\t1\t+\n", "Invalid record name"),
    ]
    for row, match in cases:
        bed6 = tmp_path / "bad.bed6"
        bed6.write_text(row)
        out = tmp_path / "out.fa"
        ge = GenomicElements(str(bed6), "bed6", str(TINY_FA))
        try:
            with pytest.raises(ValueError, match=match):
                ge.export_exogenous_sequences(str(out), record_id="name")
            assert not out.exists()
        finally:
            ge.close()


@pytest.mark.parametrize(
    "region_file_type,row",
    [
        ("bed6", "chrA\t0\t4\tpeakA\t1\t+\n"),
        ("bed6gene", "chrA\t0\t4\tpeakA\t1\t+\tGENEA\n"),
        ("narrowPeak", "chrA\t0\t4\tpeakA\t1\t+\t1.0\t-1.0\t-1.0\t0\n"),
    ],
)
def test_export_exogenous_strand_capable_schemas(tmp_path, region_file_type, row):
    """Every strand-capable registered schema accepts strand orientation (SPEC005 / #13)."""
    path = tmp_path / f"regions.{region_file_type}"
    path.write_text(row)
    ge = GenomicElements(str(path), region_file_type, str(TINY_FA))
    try:
        out = tmp_path / "out.fa"
        ge.export_exogenous_sequences(str(out), output_orientation="strand")
        assert out.read_text() == ">chrA:0-4\nACGT\n"
    finally:
        ge.close()


@pytest.mark.parametrize(
    "region_file_type,row,expected_id",
    [
        ("bed6", "chrA\t0\t4\tpeakA\t1\t+\n", "peakA"),
        ("bed6gene", "chrA\t0\t4\tpeakA\t1\t+\tGENEA\n", "peakA"),
        ("narrowPeak", "chrA\t0\t4\tpeakA\t1\t+\t1.0\t-1.0\t-1.0\t0\n", "peakA"),
        ("TREbed", "chrA\t0\t4\tpeakA\t0\t-1\n", "peakA"),
    ],
)
def test_export_exogenous_name_capable_schemas(
    tmp_path, region_file_type, row, expected_id
):
    """Every name-capable registered schema accepts name identity (SPEC005 / #13)."""
    path = tmp_path / f"regions.{region_file_type}"
    path.write_text(row)
    ge = GenomicElements(str(path), region_file_type, str(TINY_FA))
    try:
        out = tmp_path / "out.fa"
        ge.export_exogenous_sequences(str(out), record_id="name")
        assert out.read_text() == f">{expected_id}\nACGT\n"
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# Registered region types beyond bed3
# ---------------------------------------------------------------------------


def test_bedgraph_round_trip_columns(tmp_path):
    ge = GenomicElements(str(REGIONS_BEDGRAPH), "bedGraph", str(TINY_FA))
    try:
        df = ge.get_region_bed_table().to_dataframe()
        assert list(df.columns) == ["chrom", "start", "end", "dataValue"]
        assert df["dataValue"].tolist() == [1.5, 2.5]

        out = tmp_path / "roundtrip.bedGraph"
        ge.get_region_bed_table().write(str(out))

        ge2 = GenomicElements(str(out), "bedGraph", str(TINY_FA))
        try:
            df2 = ge2.get_region_bed_table().to_dataframe()
            assert list(df2.columns) == ["chrom", "start", "end", "dataValue"]
            assert df2["chrom"].tolist() == df["chrom"].tolist()
            assert df2["start"].tolist() == df["start"].tolist()
            assert df2["end"].tolist() == df["end"].tolist()
            assert df2["dataValue"].tolist() == [1.5, 2.5]
        finally:
            ge2.close()
    finally:
        ge.close()


def test_bed3gene_columns():
    ge = GenomicElements(str(REGIONS_BED3GENE), "bed3gene", str(TINY_FA))
    try:
        df = ge.get_region_bed_table().to_dataframe()
        assert list(df.columns) == ["chrom", "start", "end", "gene_symbol"]
        assert df["gene_symbol"].tolist() == ["GENE1", "GENE2"]
    finally:
        ge.close()
