"""SPEC019 contract tests: ExogeneousSequenceTools signal/stat utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from ExogeneousSequenceTools.SignalTrack import SignalTrack
from RGTools.ExogeneousSequences import ExogeneousSequences

# Tiny track for dim-reduction: shape (2, 4)
# row0: [1, 5, 2, 4] → max=5@1, min=1@0
# row1: [3, 0, 7, 1] → max=7@2, min=0@1
TINY_TRACK = np.array(
    [
        [1.0, 5.0, 2.0, 4.0],
        [3.0, 0.0, 7.0, 1.0],
    ],
    dtype=np.float64,
)


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _write_fasta(path: Path, n: int = 3) -> Path:
    ExogeneousSequences.write_sequences_to_fasta(
        [f"seq{i}" for i in range(n)],
        ["ATCG", "TTGA", "CCAT"][:n],
        str(path),
    )
    return path


# ---------------------------------------------------------------------------
# gen_track single_loc
# ---------------------------------------------------------------------------


def test_gen_track_single_loc_shape_and_fill(tmp_path: Path):
    """Build (N, 1) int64 array filled with --loc."""
    fasta = _write_fasta(tmp_path / "seqs.fa", n=3)
    out = tmp_path / "loc.npy"
    SignalTrack.gen_track_main(
        _ns(operation="single_loc", fasta=str(fasta), loc=5, output_npy=str(out))
    )
    arr = np.load(out)
    assert arr.shape == (3, 1)
    assert arr.dtype == np.int64
    assert np.array_equal(arr[:, 0], np.array([5, 5, 5]))


# ---------------------------------------------------------------------------
# track_dim_reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("max", [[5.0], [7.0]]),
        ("argmax", [[1], [2]]),
        ("min", [[1.0], [0.0]]),
        ("argmin", [[0], [1]]),
    ],
)
def test_track_dim_reduction_ops(tmp_path: Path, operation: str, expected):
    """Reduce along axis 1 with keepdims → (N, 1) for max/argmax/min/argmin."""
    inp = tmp_path / "track.npy"
    out = tmp_path / "stat.npy"
    np.save(inp, TINY_TRACK)
    SignalTrack.track_dim_reduction_main(
        _ns(
            operation=operation,
            input_npy=str(inp),
            output_npy=str(out),
            search_range=None,
        )
    )
    arr = np.load(out)
    assert arr.shape == (2, 1)
    np.testing.assert_array_equal(arr, np.array(expected))


def test_track_dim_reduction_search_range_masks_outside(tmp_path: Path):
    """Mask columns outside [start, end) with -inf before reduction."""
    # Global argmax of row0 is col 1 (5); of row1 is col 2 (7).
    # With range [0, 2): only cols 0,1 — row1 max becomes 3@0 (col 2 masked).
    inp = tmp_path / "track.npy"
    out = tmp_path / "stat.npy"
    np.save(inp, TINY_TRACK)
    SignalTrack.track_dim_reduction_main(
        _ns(
            operation="argmax",
            input_npy=str(inp),
            output_npy=str(out),
            search_range="0,2",
        )
    )
    arr = np.load(out)
    assert arr.shape == (2, 1)
    np.testing.assert_array_equal(arr, np.array([[1], [0]]))

    SignalTrack.track_dim_reduction_main(
        _ns(
            operation="max",
            input_npy=str(inp),
            output_npy=str(out),
            search_range="0,2",
        )
    )
    arr = np.load(out)
    np.testing.assert_array_equal(arr, np.array([[5.0], [3.0]]))


# ---------------------------------------------------------------------------
# print_stat
# ---------------------------------------------------------------------------


def test_print_stat_one_value_per_line(tmp_path: Path, capsys):
    """Print each column-0 value, one per line, for shape (N, 1)."""
    inp = tmp_path / "stat.npy"
    np.save(inp, np.array([[1], [2], [3]], dtype=np.int64))
    SignalTrack.print_stat_main(_ns(input_npy=str(inp)))
    assert capsys.readouterr().out == "1\n2\n3\n"


def test_print_stat_rejects_second_dim_not_one(tmp_path: Path):
    """ValueError when shape[1] != 1."""
    inp = tmp_path / "bad.npy"
    np.save(inp, np.arange(6).reshape(3, 2))
    with pytest.raises(ValueError, match="second dimension of the region stat must be 1"):
        SignalTrack.print_stat_main(_ns(input_npy=str(inp)))
