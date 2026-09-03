"""SPEC014 contract tests: GenomicElementTools import / export."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from GenomicElementTools.cli import GenomicElementTools

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "get"
TINY_BED3 = FIXTURES / "tiny.bed3"
STATS_LIST = FIXTURES / "stats.list"
CHROM_SIZES = FIXTURES / "chrom.sizes"
RSID_BED6 = FIXTURES / "rsid.bed6"


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def test_stat_list_import_export_round_trip(tmp_path: Path):
    """import/export stat_list round-trip preserves values and region order (SPEC014)."""
    npy_path = tmp_path / "stats.npy"
    list_out = tmp_path / "roundtrip.list"

    _run_cli(
        [
            "import",
            "stat_list",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--inpath",
            str(STATS_LIST),
            "--opath",
            str(npy_path),
            "--dtype",
            "np.float64",
        ]
    )

    loaded = np.load(npy_path).ravel()
    assert loaded.shape == (3,)
    assert np.allclose(loaded, [1.5, 2.5, 3.5])

    _run_cli(
        [
            "export",
            "stat_list",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--stat_npy",
            str(npy_path),
            "--opath",
            str(list_out),
            "--dtype",
            "np.float64",
        ]
    )

    assert list_out.read_text().splitlines() == ["1.5", "2.5", "3.5"]


def test_import_stat_list_rejects_non_npy_npz_suffix(tmp_path: Path):
    """Other output suffixes → ValueError (SPEC014)."""
    with pytest.raises(ValueError, match="Invalid output file type"):
        _run_cli(
            [
                "import",
                "stat_list",
                "--region_file_path",
                str(TINY_BED3),
                "--region_file_type",
                "bed3",
                "--inpath",
                str(STATS_LIST),
                "--opath",
                str(tmp_path / "stats.txt"),
                "--dtype",
                "np.float64",
            ]
        )


def test_import_stat_list_length_mismatch(tmp_path: Path):
    """List length must equal region count (SPEC014)."""
    short_list = tmp_path / "short.list"
    short_list.write_text("1.0\n2.0\n")
    with pytest.raises(ValueError, match="does not match"):
        _run_cli(
            [
                "import",
                "stat_list",
                "--region_file_path",
                str(TINY_BED3),
                "--region_file_type",
                "bed3",
                "--inpath",
                str(short_list),
                "--opath",
                str(tmp_path / "out.npy"),
                "--dtype",
                "np.float64",
            ]
        )


def test_export_chrom_filtered_ge(tmp_path: Path):
    """Keep regions whose chrom appears in chrom-size file (SPEC014 ChromFilteredGE)."""
    out = tmp_path / "filtered.bed3"
    _run_cli(
        [
            "export",
            "ChromFilteredGE",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--chrom_size",
            str(CHROM_SIZES),
            "--opath",
            str(out),
        ]
    )
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert lines == ["chrA\t0\t10", "chrB\t5\t15"]


def test_export_masked_ge(tmp_path: Path):
    """Apply boolean mask filter; write filtered regions (SPEC014 MaskedGE)."""
    mask_path = tmp_path / "keep.npy"
    np.save(mask_path, np.array([True, False, True], dtype=bool))
    out = tmp_path / "masked.bed3"

    _run_cli(
        [
            "export",
            "MaskedGE",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--mask_npy",
            str(mask_path),
            "--opath",
            str(out),
        ]
    )
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert lines == ["chrA\t0\t10", "chrC\t20\t30"]


def test_export_masked_ge_with_stat_annotation(tmp_path: Path):
    """MaskedGE optionally filters parallel annotations (SPEC014)."""
    mask_path = tmp_path / "keep.npy"
    np.save(mask_path, np.array([True, False, True], dtype=bool))
    stat_path = tmp_path / "score.npy"
    np.save(stat_path, np.array([10.0, 20.0, 30.0]))
    out_bed = tmp_path / "masked.bed3"
    anno_header = tmp_path / "masked_anno"

    _run_cli(
        [
            "export",
            "MaskedGE",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--mask_npy",
            str(mask_path),
            "--opath",
            str(out_bed),
            "--anno_name",
            "score",
            "--anno_npy",
            str(stat_path),
            "--anno_type",
            "stat",
            "--anno_oheader",
            str(anno_header),
        ]
    )

    filtered_stat = np.load(f"{anno_header}.score.npy")
    assert np.allclose(filtered_stat.ravel(), [10.0, 30.0])


def test_export_bed6poly_mocked_ensembl(tmp_path: Path):
    """bed6poly resolves rsids via Ensembl; mock REST so no network (SPEC014)."""
    out = tmp_path / "poly.bed6"

    def fake_info(self, rsid: str):
        if rsid == "rsTest1":
            return {
                "chrom": "chr1",
                "start": 100,
                "end": 101,
                "bases": "A/G",
            }
        raise Exception(f"rsid not found: {rsid}")

    with patch(
        "GenomicElementTools.export.EnsemblRestSearch.get_rsid_snp_simple_info",
        fake_info,
    ):
        _run_cli(
            [
                "export",
                "bed6poly",
                "--region_file_path",
                str(RSID_BED6),
                "--region_file_type",
                "bed6",
                "--genome_version",
                "hg19",
                "--rsid_not_found_handling",
                "drop",
                "--opath",
                str(out),
            ]
        )

    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].endswith("A/G")
    assert "rsTest1" in lines[0]


def test_export_exogenous_sequences_rejects_out_of_bounds_before_publish(tmp_path: Path):
    """CLI export ExogenousSequences validates full containment before writing (SPEC014)."""
    tiny_fa = Path(__file__).resolve().parents[1] / "fixtures" / "spec" / "tiny.fa"
    bed = tmp_path / "past.bed3"
    bed.write_text("chrA\t0\t4\nchrA\t0\t9\n")
    out = tmp_path / "out.fa"

    with pytest.raises(ValueError, match=r"chrA:0-9"):
        _run_cli(
            [
                "export",
                "ExogenousSequences",
                "--fasta_path",
                str(tiny_fa),
                "--region_file_path",
                str(bed),
                "--region_file_type",
                "bed3",
                "--opath",
                str(out),
            ]
        )
    assert not out.exists()
