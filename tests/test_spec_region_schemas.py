"""Black-box region-schema loading through GenomicElements (SPEC002, SPEC004, SPEC005)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RGTools import GenomicElements
from RGTools.BedTable import BedTable3, BedTable3Plus, BedTable6, BedTable6Plus
from RGTools.exceptions import BedTableLoadException

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spec"
TINY_FA = FIXTURES / "tiny.fa"
REGIONS_BED3 = FIXTURES / "regions.bed3"
REGIONS_BEDGRAPH = FIXTURES / "regions.bedGraph"
REGIONS_BED3GENE = FIXTURES / "regions.bed3gene"


def _write_schema(path: Path, *, base_type: str, extra_columns: list) -> Path:
    payload = {
        "schema_version": 1,
        "base_type": base_type,
        "extra_columns": extra_columns,
    }
    path.write_text(json.dumps(payload))
    return path


def _write_bed(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return path


# ---------------------------------------------------------------------------
# Named formats remain compatible and resolve as Plus schemas (SPEC002/SPEC005)
# ---------------------------------------------------------------------------


def test_spec005_named_bed3_constructor_resolves_bed3plus_schema():
    ge = GenomicElements(str(REGIONS_BED3), "bed3", str(TINY_FA))
    try:
        assert ge.region_file_type == "bed3"
        assert ge.fasta_path == str(TINY_FA)
        assert ge.region_file_path == str(REGIONS_BED3)
        assert ge.get_num_regions() == 2
        bt = ge.get_region_bed_table()
        assert isinstance(bt, BedTable3Plus)
        assert bt.extra_column_names == []
        assert bt.get_chrom_names().tolist() == ["chrB", "chrA"]
    finally:
        ge.close()


def test_spec005_named_bed6_constructor_resolves_bed6plus_schema(tmp_path):
    bed = _write_bed(
        tmp_path / "regions.bed6",
        ["chrA\t0\t4\tn1\t1.0\t+", "chrB\t1\t5\tn2\t2.0\t-"],
    )
    ge = GenomicElements(str(bed), "bed6", str(TINY_FA))
    try:
        assert ge.region_file_type == "bed6"
        bt = ge.get_region_bed_table()
        assert isinstance(bt, BedTable6Plus)
        assert bt.extra_column_names == []
        assert bt.get_region_names().tolist() == ["n1", "n2"]
        assert bt.get_region_strands().tolist() == ["+", "-"]
    finally:
        ge.close()


@pytest.mark.parametrize(
    "region_file_type,fixture,extra_names",
    [
        ("bed3gene", REGIONS_BED3GENE, ["gene_symbol"]),
        ("bedGraph", REGIONS_BEDGRAPH, ["dataValue"]),
    ],
)
def test_spec005_named_plus_formats_remain_compatible(
    region_file_type, fixture, extra_names
):
    ge = GenomicElements(str(fixture), region_file_type, str(TINY_FA))
    try:
        assert ge.region_file_type == region_file_type
        bt = ge.get_region_bed_table()
        assert isinstance(bt, BedTable3Plus)
        assert bt.extra_column_names == extra_names
        assert ge.get_num_regions() == 2
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# Custom version-1 schemas (SPEC002/SPEC005)
# ---------------------------------------------------------------------------


def test_spec005_custom_bed3plus_schema_loads_typed_extras(tmp_path):
    schema = _write_schema(
        tmp_path / "meta.region.json",
        base_type="bed3",
        extra_columns=[
            {"name": "label", "dtype": "str"},
            {"name": "count", "dtype": "int"},
            {"name": "score", "dtype": "float"},
        ],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        [
            "chrB\t1\t5\tpeakB\t3\t1.5",
            "chrA\t0\t4\tpeakA\t.\t2.25",
        ],
    )
    ge = GenomicElements(str(bed), str(schema), str(TINY_FA))
    try:
        assert ge.region_file_type == str(schema)
        assert ge.get_num_regions() == 2
        bt = ge.get_region_bed_table()
        assert isinstance(bt, BedTable3Plus)
        assert bt.extra_column_names == ["label", "count", "score"]
        assert bt.extra_column_dtype == [str, int, float]
        assert bt.get_chrom_names().tolist() == ["chrB", "chrA"]
        assert bt.get_region_extra_column("label").tolist() == ["peakB", "peakA"]
        counts = bt.get_region_extra_column("count")
        assert counts[0] == 3
        assert pd.isna(counts[1])
        assert bt.get_region_extra_column("score").tolist() == pytest.approx([1.5, 2.25])
    finally:
        ge.close()


def test_spec005_custom_bed6plus_schema_and_empty_extras(tmp_path):
    schema_empty = _write_schema(
        tmp_path / "plain.bed6.json",
        base_type="bed6",
        extra_columns=[],
    )
    bed = _write_bed(
        tmp_path / "regions.bed6",
        ["chrA\t0\t4\tn1\t1.0\t+", "chrB\t1\t5\tn2\t.\t-"],
    )
    ge = GenomicElements(str(bed), str(schema_empty), str(TINY_FA))
    try:
        bt = ge.get_region_bed_table()
        assert isinstance(bt, BedTable6Plus)
        assert bt.extra_column_names == []
        assert bt.get_region_names().tolist() == ["n1", "n2"]
        scores = bt.get_region_scores()
        assert scores[0] == pytest.approx(1.0)
        assert np.isnan(scores[1])
    finally:
        ge.close()

    schema_plus = _write_schema(
        tmp_path / "meta.bed6.json",
        base_type="bed6",
        extra_columns=[{"name": "note", "dtype": "str"}],
    )
    bed_plus = _write_bed(
        tmp_path / "regions_plus.bed6",
        ["chrA\t0\t4\tn1\t1.0\t+\thello", "chrB\t1\t5\tn2\t2.0\t-\t."],
    )
    ge2 = GenomicElements(str(bed_plus), str(schema_plus), str(TINY_FA))
    try:
        bt2 = ge2.get_region_bed_table()
        assert isinstance(bt2, BedTable6Plus)
        assert bt2.extra_column_names == ["note"]
        notes = bt2.get_region_extra_column("note")
        assert notes[0] == "hello"
        assert pd.isna(notes[1])
    finally:
        ge2.close()


def test_spec002_custom_schema_missing_dot_roundtrip(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[
            {"name": "s", "dtype": "str"},
            {"name": "i", "dtype": "int"},
            {"name": "f", "dtype": "float"},
        ],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrA\t0\t4\t.\t.\t."],
    )
    ge = GenomicElements(str(bed), str(schema), str(TINY_FA))
    try:
        out = tmp_path / "roundtrip.bed"
        ge.get_region_bed_table().write(str(out))
        assert out.read_text().splitlines()[0] == "chrA\t0\t4\t.\t.\t."
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# Path resolution and precedence (SPEC005)
# ---------------------------------------------------------------------------


def test_spec005_relative_schema_resolves_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schema = _write_schema(
        tmp_path / "rel_schema.json",
        base_type="bed3",
        extra_columns=[{"name": "tag", "dtype": "str"}],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4\tx"])
    ge = GenomicElements(str(bed), "rel_schema.json", str(TINY_FA))
    try:
        assert ge.region_file_type == "rel_schema.json"
        assert ge.get_region_bed_table().get_region_extra_column("tag").tolist() == ["x"]
        assert Path("rel_schema.json").resolve() == schema.resolve()
    finally:
        ge.close()


def test_spec005_predefined_name_precedes_same_named_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # A file named "bed3" must not override the predefined bed3 schema.
    _write_schema(
        tmp_path / "bed3",
        base_type="bed3",
        extra_columns=[{"name": "shadow", "dtype": "str"}],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4"])
    ge = GenomicElements(str(bed), "bed3", str(TINY_FA))
    try:
        bt = ge.get_region_bed_table()
        assert bt.extra_column_names == []
        assert isinstance(bt, BedTable3Plus)
    finally:
        ge.close()


def test_spec005_explicit_relative_path_selects_shadowed_schema_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_schema(
        tmp_path / "bed3",
        base_type="bed3",
        extra_columns=[{"name": "shadow", "dtype": "str"}],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4\tkept"])
    ge = GenomicElements(str(bed), "./bed3", str(TINY_FA))
    try:
        bt = ge.get_region_bed_table()
        assert bt.extra_column_names == ["shadow"]
        assert bt.get_region_extra_column("shadow").tolist() == ["kept"]
    finally:
        ge.close()


def test_spec005_canonical_paths_identify_same_schema_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4"])
    rel = GenomicElements(str(bed), "schema.json", str(TINY_FA))
    abs_ge = GenomicElements(str(bed), str(schema.resolve()), str(TINY_FA))
    try:
        assert Path(rel.region_file_type).resolve() == Path(abs_ge.region_file_type).resolve()
        # Public identity for same canonical file is observable via resolved path equality.
        assert schema.resolve() == Path("schema.json").resolve()
    finally:
        rel.close()
        abs_ge.close()


# ---------------------------------------------------------------------------
# Row validation (SPEC002/SPEC004)
# ---------------------------------------------------------------------------


def test_spec004_exact_column_count_required_for_custom_schema(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[{"name": "tag", "dtype": "str"}],
    )
    too_few = _write_bed(tmp_path / "few.bed", ["chrA\t0\t4"])
    too_many = _write_bed(tmp_path / "many.bed", ["chrA\t0\t4\ta\tb"])
    with pytest.raises(BedTableLoadException, match="column"):
        GenomicElements(str(too_few), str(schema), str(TINY_FA))
    with pytest.raises(BedTableLoadException, match="column"):
        GenomicElements(str(too_many), str(schema), str(TINY_FA))


def test_spec004_unparseable_extra_value_raises_before_usable_collection(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[{"name": "count", "dtype": "int"}],
    )
    bed = _write_bed(tmp_path / "bad.bed", ["chrA\t0\t4\tnot-an-int"])
    with pytest.raises((ValueError, BedTableLoadException, TypeError)):
        GenomicElements(str(bed), str(schema), str(TINY_FA))


# ---------------------------------------------------------------------------
# Schema validation (SPEC002/SPEC005)
# ---------------------------------------------------------------------------


def test_spec005_unknown_selector_reports_neither_named_nor_readable(tmp_path):
    with pytest.raises(ValueError, match="neither a supported named format nor a readable"):
        GenomicElements(str(REGIONS_BED3), "not_a_type", str(TINY_FA))


@pytest.mark.parametrize(
    "payload,match",
    [
        ("{not json", "JSON"),
        ({"schema_version": True, "base_type": "bed3", "extra_columns": []}, "schema_version"),
        ({"schema_version": 2, "base_type": "bed3", "extra_columns": []}, "schema_version"),
        ({"schema_version": 1.0, "base_type": "bed3", "extra_columns": []}, "schema_version"),
        ({"schema_version": 1, "base_type": "bed12", "extra_columns": []}, "base_type"),
        ({"schema_version": 1, "extra_columns": []}, "base_type"),
        ({"schema_version": 1, "base_type": "bed3", "extra_columns": [], "extra": 1}, "field"),
        ({"schema_version": 1, "base_type": "bed3", "extra_columns": "nope"}, "extra_columns"),
        (
            {
                "schema_version": 1,
                "base_type": "bed3",
                "extra_columns": [{"name": "a", "dtype": "str", "extra": 1}],
            },
            "extra",
        ),
        (
            {
                "schema_version": 1,
                "base_type": "bed3",
                "extra_columns": [{"name": "a", "dtype": "bool"}],
            },
            "dtype",
        ),
        (
            {
                "schema_version": 1,
                "base_type": "bed3",
                "extra_columns": [{"name": "", "dtype": "str"}],
            },
            "name",
        ),
        (
            {
                "schema_version": 1,
                "base_type": "bed3",
                "extra_columns": [{"name": "a\tb", "dtype": "str"}],
            },
            "name",
        ),
        (
            {
                "schema_version": 1,
                "base_type": "bed3",
                "extra_columns": [
                    {"name": "dup", "dtype": "str"},
                    {"name": "dup", "dtype": "int"},
                ],
            },
            "dup",
        ),
        (
            {
                "schema_version": 1,
                "base_type": "bed3",
                "extra_columns": [{"name": "chrom", "dtype": "str"}],
            },
            "chrom",
        ),
        (
            {
                "schema_version": 1,
                "base_type": "bed6",
                "extra_columns": [{"name": "strand", "dtype": "str"}],
            },
            "strand",
        ),
    ],
)
def test_spec002_schema_validation_rejects_malformed_selectors(
    tmp_path, payload, match
):
    path = tmp_path / "bad_schema.json"
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        GenomicElements(str(REGIONS_BED3), str(path), str(TINY_FA))


def test_spec002_meaningful_names_with_spaces_and_punctuation_accepted(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[{"name": "gene symbol (ref)", "dtype": "str"}],
    )
    bed = _write_bed(tmp_path / "regions.bed", ["chrA\t0\t4\tTP53"])
    ge = GenomicElements(str(bed), str(schema), str(TINY_FA))
    try:
        bt = ge.get_region_bed_table()
        assert bt.extra_column_names == ["gene symbol (ref)"]
        assert bt.get_region_extra_column("gene symbol (ref)").tolist() == ["TP53"]
    finally:
        ge.close()


# ---------------------------------------------------------------------------
# Snapshot lifetime and derived collections (SPEC005)
# ---------------------------------------------------------------------------


def test_spec005_filter_preserves_custom_schema_and_annotations(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[{"name": "tag", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrB\t1\t5\tkeep", "chrA\t0\t4\tdrop"],
    )
    ge = GenomicElements(str(bed), str(schema), str(TINY_FA))
    try:
        ge.load_region_stat_from_arr("score", np.array([10.0, 20.0]))
        out = tmp_path / "filtered.bed"
        filtered = ge.apply_logical_filter(np.array([True, False]), str(out))
        try:
            bt = filtered.get_region_bed_table()
            assert bt.extra_column_names == ["tag"]
            assert bt.extra_column_dtype == [str]
            assert bt.get_chrom_names().tolist() == ["chrB"]
            assert bt.get_region_extra_column("tag").tolist() == ["keep"]
            assert filtered.get_stat_arr("score").ravel().tolist() == [10.0]
            assert filtered.region_file_type == str(schema)
        finally:
            filtered.close()
    finally:
        ge.close()


def test_spec005_schema_snapshot_survives_schema_file_deletion(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[{"name": "tag", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrB\t1\t5\tkeep", "chrA\t0\t4\tdrop"],
    )
    ge = GenomicElements(str(bed), str(schema), str(TINY_FA))
    try:
        schema.unlink()
        out = tmp_path / "filtered.bed"
        filtered = ge.apply_logical_filter(np.array([True, False]), str(out))
        try:
            bt = filtered.get_region_bed_table()
            assert bt.extra_column_names == ["tag"]
            assert bt.get_region_extra_column("tag").tolist() == ["keep"]
        finally:
            filtered.close()
    finally:
        ge.close()


def test_spec005_schema_snapshot_survives_schema_file_change(tmp_path):
    schema = _write_schema(
        tmp_path / "schema.json",
        base_type="bed3",
        extra_columns=[{"name": "tag", "dtype": "str"}],
    )
    bed = _write_bed(
        tmp_path / "regions.bed",
        ["chrA\t0\t4\told"],
    )
    ge = GenomicElements(str(bed), str(schema), str(TINY_FA))
    try:
        _write_schema(
            schema,
            base_type="bed3",
            extra_columns=[
                {"name": "tag", "dtype": "str"},
                {"name": "extra", "dtype": "int"},
            ],
        )
        out = tmp_path / "filtered.bed"
        filtered = ge.apply_logical_filter(np.array([True]), str(out))
        try:
            bt = filtered.get_region_bed_table()
            assert bt.extra_column_names == ["tag"]
            assert "extra" not in bt.extra_column_names
        finally:
            filtered.close()
    finally:
        ge.close()


def test_spec005_failed_schema_validation_creates_no_output(tmp_path):
    dest = tmp_path / "should_not_exist.bed"
    bad_schema = tmp_path / "bad.json"
    bad_schema.write_text("{")
    assert not dest.exists()
    with pytest.raises(ValueError):
        GenomicElements(str(REGIONS_BED3), str(bad_schema), str(TINY_FA))
    assert not dest.exists()


# ---------------------------------------------------------------------------
# Direct BedTable APIs remain green (SPEC004)
# ---------------------------------------------------------------------------


def test_spec004_direct_bedtable_construction_apis_remain_supported(tmp_path):
    b3 = BedTable3(enable_sort=False)
    b3.load_from_file(str(REGIONS_BED3))
    assert len(b3) == 2

    b6 = BedTable6(enable_sort=False)
    bed6 = _write_bed(tmp_path / "r.bed6", ["chrA\t0\t4\tn\t1.0\t+"])
    b6.load_from_file(str(bed6))
    assert list(b6.get_region_names()) == ["n"]

    b3p = BedTable3Plus(extra_column_names=["g"], extra_column_dtype=[str], enable_sort=False)
    bed3p = _write_bed(tmp_path / "r.bed3p", ["chrA\t0\t4\tTP53"])
    b3p.load_from_file(str(bed3p))
    assert list(b3p.get_region_extra_column("g")) == ["TP53"]

    b6p = BedTable6Plus(extra_column_names=["g"], extra_column_dtype=[str], enable_sort=False)
    bed6p = _write_bed(tmp_path / "r.bed6p", ["chrA\t0\t4\tn\t1.0\t+\tTP53"])
    b6p.load_from_file(str(bed6p))
    assert list(b6p.get_region_extra_column("g")) == ["TP53"]
