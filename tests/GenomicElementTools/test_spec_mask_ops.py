"""SPEC015 contract tests: GenomicElementTools mask_op (intersect / union / opposite)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from GenomicElementTools.cli import GenomicElementTools

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "get"
TINY_BED3 = FIXTURES / "tiny.bed3"


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _write_mask(path: Path, values: list[bool]) -> Path:
    np.save(path, np.asarray(values, dtype=bool))
    return path


def test_mask_op_intersect_and_union(tmp_path: Path):
    """Element-wise AND / OR over masks aligned to tiny bed3 (SPEC015)."""
    m1 = _write_mask(tmp_path / "m1.npy", [True, True, False])
    m2 = _write_mask(tmp_path / "m2.npy", [True, False, True])
    intersect_out = tmp_path / "intersect.npy"
    union_out = tmp_path / "union.npy"

    _run_cli(
        [
            "mask_op",
            "intersect",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--mask_npy",
            str(m1),
            "--mask_npy",
            str(m2),
            "--opath",
            str(intersect_out),
        ]
    )
    _run_cli(
        [
            "mask_op",
            "union",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--mask_npy",
            str(m1),
            "--mask_npy",
            str(m2),
            "--opath",
            str(union_out),
        ]
    )

    assert np.array_equal(
        np.load(intersect_out).ravel(), np.array([True, False, False])
    )
    assert np.array_equal(np.load(union_out).ravel(), np.array([True, True, True]))


def test_mask_op_opposite(tmp_path: Path):
    """Element-wise NOT of a single mask (SPEC015)."""
    mask = _write_mask(tmp_path / "mask.npy", [True, False, True])
    out = tmp_path / "opposite.npy"

    _run_cli(
        [
            "mask_op",
            "opposite",
            "--region_file_path",
            str(TINY_BED3),
            "--region_file_type",
            "bed3",
            "--mask_npy",
            str(mask),
            "--opath",
            str(out),
        ]
    )

    assert np.array_equal(np.load(out).ravel(), np.array([False, True, False]))


@pytest.mark.parametrize("operation", ["intersect", "union"])
def test_mask_op_requires_at_least_two_masks(tmp_path: Path, operation: str):
    """intersect/union with fewer than two --mask_npy inputs → ValueError (SPEC015)."""
    mask = _write_mask(tmp_path / "only.npy", [True, False, True])
    out = tmp_path / "out.npy"

    with pytest.raises(ValueError, match="[Aa]t least two"):
        _run_cli(
            [
                "mask_op",
                operation,
                "--region_file_path",
                str(TINY_BED3),
                "--region_file_type",
                "bed3",
                "--mask_npy",
                str(mask),
                "--opath",
                str(out),
            ]
        )
