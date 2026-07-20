"""Spec-derived tests for BedTable + region TSV contracts (SPEC002 + SPEC004)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RGTools.BedTable import (
    BedRegion,
    BedTable3,
    BedTable3Plus,
    BedTable6,
    BedTable6Plus,
    BedTablePairEnd,
)
from RGTools.exceptions import (
    BedTableLoadException,
    InvalidBedRegionException,
    InvalidStrandnessException,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spec"


# ---------------------------------------------------------------------------
# BedRegion (SPEC004)
# ---------------------------------------------------------------------------


class TestBedRegion:
    def test_equality_uses_only_chrom_start_end(self):
        a = BedRegion("chr1", 10, 20, name="a", strand="+")
        b = BedRegion("chr1", 10, 20, name="b", strand="-")
        c = BedRegion("chr1", 10, 21)
        assert a == b
        assert a != c

    def test_dict_like_field_access(self):
        r = BedRegion("chr1", 10, 20, name="peak", strand="+")
        assert r["chrom"] == "chr1"
        assert r["start"] == 10
        assert r["end"] == 20
        assert r["name"] == "peak"
        assert r["strand"] == "+"

    def test_pad_region_returns_new_instance_plus_strand(self):
        r = BedRegion("chr1", 10, 20, strand="+")
        padded = r.pad_region(5, 3)
        assert padded is not r
        assert (padded["start"], padded["end"]) == (5, 23)
        assert (r["start"], r["end"]) == (10, 20)

    def test_pad_region_minus_strand_swaps_upstream_downstream(self):
        r = BedRegion("chr1", 10, 20, strand="-")
        padded = r.pad_region(5, 3)
        # upstream=5 extends end; downstream=3 extends start (relative to -)
        assert (padded["start"], padded["end"]) == (7, 25)

    def test_pad_region_ignore_strand_treats_as_plus(self):
        r = BedRegion("chr1", 10, 20)  # no strand
        padded = r.pad_region(5, 3, ignore_strand=True)
        assert (padded["start"], padded["end"]) == (5, 23)

    def test_pad_region_missing_strand_raises(self):
        with pytest.raises(InvalidStrandnessException):
            BedRegion("chr1", 10, 20).pad_region(1, 1)

    @pytest.mark.parametrize("strand", [".", "*", "forward"])
    def test_pad_region_invalid_strand_raises(self, strand):
        with pytest.raises(InvalidStrandnessException):
            BedRegion("chr1", 10, 20, strand=strand).pad_region(1, 1)

    def test_pad_region_start_ge_end_raises(self):
        with pytest.raises(InvalidBedRegionException):
            BedRegion("chr1", 10, 20, strand="+").pad_region(0, -15)

    def test_pad_region_negative_start_raises(self):
        with pytest.raises(InvalidBedRegionException):
            BedRegion("chr1", 5, 10, strand="+").pad_region(10, 0)


# ---------------------------------------------------------------------------
# BedTable3 / BedTable6 I/O (SPEC002 + SPEC004)
# ---------------------------------------------------------------------------


class TestBedTableIO:
    def test_bed3_load_write_tsv_no_header(self, tmp_path):
        bt = BedTable3(enable_sort=False)
        bt.load_from_file(str(FIXTURES / "unsorted.bed3"))
        assert len(bt) == 3
        assert list(bt.get_chrom_names()) == ["chr2", "chr1", "chr1"]
        assert list(bt.get_start_locs()) == [10, 5, 1]
        assert list(bt.get_end_locs()) == [20, 15, 2]

        out = tmp_path / "out.bed3"
        bt.write(str(out))
        text = out.read_text()
        assert not text.startswith("chrom")
        assert text.splitlines()[0] == "chr2\t10\t20"
        assert "\t" in text

    def test_bed6_missing_dot_roundtrip(self, tmp_path):
        bt = BedTable6(enable_sort=False)
        bt.load_from_file(str(FIXTURES / "sample.bed6"))
        assert len(bt) == 3

        names = list(bt.get_region_names())
        scores = list(bt.get_region_scores())
        strands = list(bt.get_region_strands())
        assert names[0] == "peakA"
        assert scores[0] == pytest.approx(12.5)
        assert strands[0] == "+"
        assert pd.isna(names[1])
        assert pd.isna(scores[1])
        assert strands[1] == "-"

        out = tmp_path / "out.bed6"
        bt.write(str(out))
        lines = out.read_text().splitlines()
        assert lines[1] == "chr1\t30\t40\t.\t.\t-"
        assert not out.read_text().startswith("chrom")

    def test_sort_toggle_on_load(self):
        sorted_bt = BedTable3(enable_sort=True)
        sorted_bt.load_from_file(str(FIXTURES / "unsorted.bed3"))
        assert list(zip(sorted_bt.get_chrom_names(), sorted_bt.get_start_locs())) == [
            ("chr1", 1),
            ("chr1", 5),
            ("chr2", 10),
        ]

        unsorted_bt = BedTable3(enable_sort=False)
        unsorted_bt.load_from_file(str(FIXTURES / "unsorted.bed3"))
        assert list(zip(unsorted_bt.get_chrom_names(), unsorted_bt.get_start_locs())) == [
            ("chr2", 10),
            ("chr1", 5),
            ("chr1", 1),
        ]

    def test_write_stdout(self):
        bt = BedTable3(enable_sort=False)
        bt.load_from_dataframe(
            pd.DataFrame({"chrom": ["chr1"], "start": [1], "end": [2]})
        )
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            bt.write("stdout")
        finally:
            sys.stdout = old
        assert buf.getvalue() == "chr1\t1\t2\n"

    def test_load_from_dataframe_column_mismatch_raises(self):
        bt = BedTable3()
        bad = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        with pytest.raises(BedTableLoadException):
            bt.load_from_dataframe(bad)

    def test_empty_table_starts_with_zero_rows(self):
        assert len(BedTable3()) == 0
        assert len(BedTable6()) == 0


# ---------------------------------------------------------------------------
# Logical filter + containment subset (SPEC004)
# ---------------------------------------------------------------------------


class TestBedTableFilterAndSubset:
    @pytest.fixture
    def unsorted_table(self):
        bt = BedTable3(enable_sort=False)
        bt.load_from_file(str(FIXTURES / "unsorted.bed3"))
        return bt

    def test_logical_filter_keeps_true_rows_as_new_table(self, unsorted_table):
        filtered = unsorted_table.apply_logical_filter(
            np.array([True, False, True])
        )
        assert filtered is not unsorted_table
        assert len(filtered) == 2
        assert list(zip(filtered.get_chrom_names(), filtered.get_start_locs())) == [
            ("chr2", 10),
            ("chr1", 1),
        ]
        assert len(unsorted_table) == 3

    def test_logical_filter_wrong_length_raises(self, unsorted_table):
        with pytest.raises(ValueError):
            unsorted_table.apply_logical_filter(np.array([True, False]))

    def test_logical_filter_non_boolean_raises(self, unsorted_table):
        with pytest.raises(ValueError):
            unsorted_table.apply_logical_filter(np.array([1, 0, 1]))

    def test_region_subset_full_containment_only(self, unsorted_table):
        # regions: chr2:10-20, chr1:5-15, chr1:1-2
        contained = unsorted_table.region_subset("chr1", 0, 20)
        assert list(
            zip(
                contained.get_chrom_names(),
                contained.get_start_locs(),
                contained.get_end_locs(),
            )
        ) == [("chr1", 5, 15), ("chr1", 1, 2)]

        # overlapping but not fully contained → excluded
        strict = unsorted_table.region_subset("chr1", 6, 14)
        assert len(strict) == 0

        exact = unsorted_table.region_subset("chr1", 5, 15)
        assert list(zip(exact.get_start_locs(), exact.get_end_locs())) == [(5, 15)]


# ---------------------------------------------------------------------------
# BedTable6 from BedTable3 (SPEC004)
# ---------------------------------------------------------------------------


class TestBedTable6FromBed3:
    def test_load_from_bedtable3_writes_dot_missing_fields(self, tmp_path):
        b3 = BedTable3(enable_sort=False)
        b3.load_from_file(str(FIXTURES / "unsorted.bed3"))
        b6 = BedTable6(enable_sort=False)
        b6.load_from_BedTable3(b3)
        assert len(b6) == 3

        out = tmp_path / "from3.bed6"
        b6.write(str(out))
        for line in out.read_text().splitlines():
            cols = line.split("\t")
            assert len(cols) == 6
            assert cols[3] == "."
            assert cols[4] == "."
            assert cols[5] == "."


# ---------------------------------------------------------------------------
# Plus extras (SPEC002 + SPEC004)
# ---------------------------------------------------------------------------


class TestBedTablePlus:
    def test_bed3plus_extra_column_load_and_accessor(self):
        bt = BedTable3Plus(extra_column_names=["gene"], enable_sort=False)
        bt.load_from_file(str(FIXTURES / "sample.bed3plus"))
        assert list(bt.get_region_extra_column("gene")) == ["TP53", "BRCA1"]
        assert list(bt.get_chrom_names()) == ["chr1", "chr1"]

    def test_bed3plus_int_extra_missing_dot(self, tmp_path):
        path = tmp_path / "plus_int.bed"
        path.write_text("chr1\t1\t2\t5\nchr1\t3\t4\t.\n")
        bt = BedTable3Plus(
            extra_column_names=["val"],
            extra_column_dtype=[int],
            enable_sort=False,
        )
        bt.load_from_file(str(path))
        vals = bt.get_region_extra_column("val")
        assert vals[0] == 5
        assert pd.isna(vals[1])

        out = tmp_path / "plus_int_out.bed"
        bt.write(str(out))
        assert out.read_text().splitlines()[1] == "chr1\t3\t4\t."

    def test_bed3plus_float_extra_missing_dot(self, tmp_path):
        path = tmp_path / "plus_float.bed"
        path.write_text("chr1\t1\t2\t3.5\nchr1\t3\t4\t.\n")
        bt = BedTable3Plus(
            extra_column_names=["val"],
            extra_column_dtype=[float],
            enable_sort=False,
        )
        bt.load_from_file(str(path))
        vals = bt.get_region_extra_column("val")
        assert vals[0] == pytest.approx(3.5)
        assert np.isnan(vals[1])

        out = tmp_path / "plus_float_out.bed"
        bt.write(str(out))
        assert out.read_text().splitlines()[1].endswith("\t.")

    def test_bed6plus_schema(self, tmp_path):
        path = tmp_path / "b6plus.bed"
        path.write_text("chr1\t1\t2\tn1\t1.0\t+\tGENE1\n")
        bt = BedTable6Plus(
            extra_column_names=["gene"],
            extra_column_dtype=[str],
            enable_sort=False,
        )
        bt.load_from_file(str(path))
        assert list(bt.get_region_names()) == ["n1"]
        assert list(bt.get_region_strands()) == ["+"]
        assert list(bt.get_region_extra_column("gene")) == ["GENE1"]


# ---------------------------------------------------------------------------
# BedTablePairEnd — custom layout, not BEDPE (SPEC002 + SPEC004)
# ---------------------------------------------------------------------------


class TestBedTablePairEnd:
    def test_column_contract_and_always_sorts_first_mate(self):
        pe = BedTablePairEnd(extra_column_names=[])
        pe.load_from_file(str(FIXTURES / "sample.pairend"))
        assert len(pe) == 2

        cols = pe.to_dataframe().columns.tolist()
        assert cols == [
            "chrom",
            "start",
            "end",
            "chrom2",
            "start2",
            "end2",
            "name",
            "score",
            "strand",
            "strand2",
        ]

        # Always sorted by first mate (chr1:1-5 before chr1:10-20)
        assert list(zip(pe.get_chrom_names(), pe.get_start_locs(), pe.get_end_locs())) == [
            ("chr1", 1, 5),
            ("chr1", 10, 20),
        ]
        assert list(pe.get_other_region_chroms()) == ["chr3", "chr2"]
        assert list(pe.get_other_region_starts()) == [50, 30]
        assert list(pe.get_other_region_ends()) == [60, 40]
        assert list(pe.get_pair_names()) == ["p2", "p1"]
        assert list(pe.get_pair_scores()) == pytest.approx([2.0, 1.5])
        assert list(pe.get_region_strands()) == ["-", "+"]
        assert list(pe.get_other_region_strands()) == ["+", "-"]

    def test_write_roundtrip_preserves_custom_layout(self, tmp_path):
        pe = BedTablePairEnd(extra_column_names=[])
        pe.load_from_file(str(FIXTURES / "sample.pairend"))
        out = tmp_path / "pe.tsv"
        pe.write(str(out))
        lines = out.read_text().splitlines()
        assert len(lines[0].split("\t")) == 10
        assert not out.read_text().startswith("chrom")
        # first-mate sorted order on disk
        assert lines[0].startswith("chr1\t1\t5\tchr3\t50\t60")

    def test_optional_extras(self, tmp_path):
        path = tmp_path / "pe_extra.tsv"
        path.write_text("chr1\t10\t20\tchr2\t30\t40\tp1\t1.5\t+\t-\tX\n")
        pe = BedTablePairEnd(
            extra_column_names=["anno"],
            extra_column_dtype=[str],
        )
        pe.load_from_file(str(path))
        assert list(pe.get_region_extra_column("anno")) == ["X"]
        assert pe.to_dataframe().columns.tolist()[-1] == "anno"
