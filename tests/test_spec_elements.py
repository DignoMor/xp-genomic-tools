"""Spec-derived tests for RGTools Elements collections (SPEC002, SPEC005)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from RGTools import ExogeneousSequences, GenomicElements

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
# ExogeneousSequences
# ---------------------------------------------------------------------------


def test_exogeneous_sequences_synthetic_bed3_view():
    es = ExogeneousSequences(str(TINY_FA))
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


def test_exogeneous_sequences_region_file_path_not_implemented():
    es = ExogeneousSequences(str(TINY_FA))
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
# export_exogeneous_sequences
# ---------------------------------------------------------------------------


def test_export_exogeneous_sequences_header_and_refuse_existing(tmp_path):
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        out = tmp_path / "exported.fa"
        ge.export_exogeneous_sequences(str(out))
        text = out.read_text()
        assert ">chrB:1-5\n" in text
        assert ">chrA:0-4\n" in text
        assert "GGGC" in text
        assert "ACGT" in text

        with pytest.raises(ValueError, match="already exists"):
            ge.export_exogeneous_sequences(str(out))
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
