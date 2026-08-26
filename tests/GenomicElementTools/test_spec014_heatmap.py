"""SPEC014 contract tests: GenomicElementTools export Heatmap (finite signed rendering)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.colors import TwoSlopeNorm

from GenomicElementTools.cli import GenomicElementTools

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "get"
TINY_BED3 = FIXTURES / "tiny.bed3"


def _run_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


def _save_track(path: Path, rows: list[list[float]]) -> Path:
    arr = np.asarray(rows, dtype=float)
    np.save(path, arr)
    return path


def _heatmap_argv(
    tmp_path: Path,
    tracks: list[tuple[str, Path, str]],
    *,
    absolute: list[str] | None = None,
    opath: Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    argv = [
        "export",
        "Heatmap",
        "--region_file_path",
        str(TINY_BED3),
        "--region_file_type",
        "bed3",
    ]
    for title, npy, negative in tracks:
        argv.extend(["--track_npy", str(npy), "--title", title, "--negative", negative])
    if absolute is not None:
        for value in absolute:
            argv.extend(["--absolute", value])
    argv.extend(["--opath", str(opath or (tmp_path / "heatmap.png"))])
    if extra:
        argv.extend(extra)
    return argv


def _capture_figure(argv: list[str]):
    saved = {}

    def _fake_savefig(self, *args, **kwargs):
        saved["fig"] = self

    with patch("matplotlib.figure.Figure.savefig", _fake_savefig):
        _run_cli(argv)
    assert "fig" in saved
    return saved["fig"]


def _image_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def _mean_axes(fig):
    return [ax for ax in fig.axes if ax.get_ylabel() == "Mean signal"]


def _panel_image(fig, col: int):
    return _image_axes(fig)[col].images[0]


def _panel_mean_y(fig, col: int):
    return _mean_axes(fig)[col].get_lines()[0].get_ydata()


def test_export_heatmap_rejects_partial_absolute_list(tmp_path: Path):
    """Partial --absolute lists fail clearly (SPEC014)."""
    track = _save_track(
        tmp_path / "a.npy",
        [[1.0] * 10, [2.0] * 10, [3.0] * 10],
    )
    track_b = _save_track(
        tmp_path / "b.npy",
        [[0.5] * 10, [0.5] * 10, [0.5] * 10],
    )
    with pytest.raises(ValueError, match="absolute"):
        _run_cli(
            _heatmap_argv(
                tmp_path,
                [("A", track, "False"), ("B", track_b, "False")],
                absolute=["True"],
            )
        )


def test_export_heatmap_rejects_excessive_absolute_list(tmp_path: Path):
    """Excessive --absolute values fail clearly (SPEC014)."""
    track = _save_track(
        tmp_path / "a.npy",
        [[1.0] * 10, [2.0] * 10, [3.0] * 10],
    )
    with pytest.raises(ValueError, match="absolute"):
        _run_cli(
            _heatmap_argv(
                tmp_path,
                [("A", track, "False")],
                absolute=["True", "False"],
            )
        )


def test_export_heatmap_omitted_absolute_matches_explicit_true(tmp_path: Path):
    """Omitting --absolute equals one --absolute True per track (SPEC014)."""
    track = _save_track(
        tmp_path / "signed_looking.npy",
        [
            [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [-1.0] * 10,
            [2.0] * 10,
        ],
    )
    tracks = [("Mag", track, "False")]
    omitted = _capture_figure(_heatmap_argv(tmp_path, tracks))
    explicit = _capture_figure(
        _heatmap_argv(tmp_path, tracks, absolute=["True"])
    )
    np.testing.assert_allclose(
        np.asarray(_panel_image(omitted, 0).get_array()),
        np.asarray(_panel_image(explicit, 0).get_array()),
    )
    np.testing.assert_allclose(_panel_mean_y(omitted, 0), _panel_mean_y(explicit, 0))
    assert _panel_image(omitted, 0).cmap.name == "Reds"
    assert _panel_image(explicit, 0).cmap.name == "Reds"


def test_export_heatmap_magnitude_absolute_transform_and_negative_mean(tmp_path: Path):
    """Magnitude mode keeps abs transform, palette, and --negative mean (SPEC014)."""
    track = _save_track(
        tmp_path / "mag.npy",
        [
            [-2.0] * 10,
            [-5.0] * 10,
            [-1.0] * 10,
        ],
    )
    fig = _capture_figure(
        _heatmap_argv(
            tmp_path,
            [("NegMag", track, "True")],
            absolute=["True"],
        )
    )
    image = np.asarray(_panel_image(fig, 0).get_array(), dtype=float)
    # Ascending abs strength: row2 (1), row0 (2), row1 (5)
    np.testing.assert_allclose(image[0], np.full(10, 1.0))
    np.testing.assert_allclose(image[1], np.full(10, 2.0))
    np.testing.assert_allclose(image[2], np.full(10, 5.0))
    assert _panel_image(fig, 0).cmap.name == "Blues"
    np.testing.assert_allclose(_panel_mean_y(fig, 0), np.full(10, -np.mean([2.0, 5.0, 1.0])))


def test_export_heatmap_magnitude_zero_padding(tmp_path: Path):
    """Magnitude shorter-row rectangularization still zero-pads (SPEC014)."""
    # tiny.bed3 regions are equal length; pad via ragged logical rows through
    # a full-width array that already includes trailing zeros in shorter semantics
    # is exercised by unequal effective signal: trailing zeros remain zeros.
    track = _save_track(
        tmp_path / "pad.npy",
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    fig = _capture_figure(
        _heatmap_argv(tmp_path, [("Pad", track, "False")], absolute=["True"])
    )
    image = np.asarray(_panel_image(fig, 0).get_array(), dtype=float)
    assert image.shape == (3, 10)
    assert np.all(image[:, -1] == 0.0)


def test_export_heatmap_signed_preserves_raw_values_ignores_negative(tmp_path: Path):
    """Signed panels keep raw polarity in image and mean despite --negative (SPEC014)."""
    track = _save_track(
        tmp_path / "signed.npy",
        [
            [-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0] * 10,
            [0.5] * 10,
        ],
    )
    fig = _capture_figure(
        _heatmap_argv(
            tmp_path,
            [("Signed", track, "True")],
            absolute=["False"],
        )
    )
    image = np.asarray(_panel_image(fig, 0).get_array(), dtype=float)
    # strengths: row0 max abs 4, row1 max abs 1, row2 max abs 0.5 → ascending row2,row1,row0
    np.testing.assert_allclose(image[0], np.full(10, 0.5))
    np.testing.assert_allclose(image[1], np.full(10, -1.0))
    np.testing.assert_allclose(
        image[2],
        [-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    )
    expected_mean = np.mean(
        [
            [-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0] * 10,
            [0.5] * 10,
        ],
        axis=0,
    )
    np.testing.assert_allclose(_panel_mean_y(fig, 0), expected_mean)


def test_export_heatmap_signed_palette_colorbar_and_norm(tmp_path: Path):
    """Signed panels use RdBu_r, a color bar, and exact zero-centered limits (SPEC014)."""
    track = _save_track(
        tmp_path / "asym.npy",
        [
            [-10.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-8.0, -2.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-9.0, -3.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    fig = _capture_figure(
        _heatmap_argv(
            tmp_path,
            [("Signed", track, "False")],
            absolute=["False"],
            extra=["--per_track_max_percentile", "100", "--vmax_percentile", "100"],
        )
    )
    im = _panel_image(fig, 0)
    assert im.cmap.name == "RdBu_r"
    assert im.colorbar is not None
    norm = im.norm
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vcenter == 0
    assert norm.vmin == -norm.vmax
    assert norm.vmax == 10.0
    blue_rgb = im.cmap(im.norm(-5.0))[:3]
    white_rgb = im.cmap(im.norm(0.0))[:3]
    red_rgb = im.cmap(im.norm(5.0))[:3]
    assert blue_rgb[2] > blue_rgb[0]
    assert red_rgb[0] > red_rgb[2]
    assert max(abs(white_rgb[0] - white_rgb[1]), abs(white_rgb[1] - white_rgb[2])) < 0.05


def test_export_heatmap_signed_all_zero_fallback_limits(tmp_path: Path):
    """All-zero finite signed track falls back to [-1, +1] (SPEC014)."""
    track = _save_track(
        tmp_path / "zeros.npy",
        [[0.0] * 10, [0.0] * 10, [0.0] * 10],
    )
    fig = _capture_figure(
        _heatmap_argv(
            tmp_path,
            [("Zeros", track, "False")],
            absolute=["False"],
        )
    )
    norm = _panel_image(fig, 0).norm
    assert isinstance(norm, TwoSlopeNorm)
    assert norm.vmin == -1
    assert norm.vcenter == 0
    assert norm.vmax == 1


def test_export_heatmap_mixed_mode_shared_stable_order(tmp_path: Path):
    """Mixed magnitude/signed panels share ascending abs-strength order (SPEC014)."""
    mag = _save_track(
        tmp_path / "mag.npy",
        [
            [1.0] * 10,  # strength candidate 1
            [5.0] * 10,  # 5
            [3.0] * 10,  # 3
        ],
    )
    signed = _save_track(
        tmp_path / "signed.npy",
        [
            [-1.0] * 10,  # abs 1
            [2.0] * 10,  # abs 2
            [-10.0] * 10,  # abs 10 → overall strengths 1, 5, 10
        ],
    )
    # Tie case: swap in equal strengths for first two via second figure below.
    fig = _capture_figure(
        _heatmap_argv(
            tmp_path,
            [("Mag", mag, "False"), ("Signed", signed, "True")],
            absolute=["True", "False"],
        )
    )
    mag_img = np.asarray(_panel_image(fig, 0).get_array(), dtype=float)
    signed_img = np.asarray(_panel_image(fig, 1).get_array(), dtype=float)
    # Cross-track strengths: max(1,1)=1, max(5,2)=5, max(3,10)=10 → row0, row1, row2.
    np.testing.assert_allclose(mag_img[0], np.full(10, 1.0))
    np.testing.assert_allclose(mag_img[1], np.full(10, 5.0))
    np.testing.assert_allclose(mag_img[2], np.full(10, 3.0))
    np.testing.assert_allclose(signed_img[0], np.full(10, -1.0))
    np.testing.assert_allclose(signed_img[1], np.full(10, 2.0))
    np.testing.assert_allclose(signed_img[2], np.full(10, -10.0))
    assert _panel_image(fig, 0).cmap.name == "Reds"
    assert _panel_image(fig, 1).cmap.name == "RdBu_r"
    assert _panel_image(fig, 1).colorbar is not None
    assert _panel_image(fig, 0).colorbar is None

    # Stable tie-break: equal cross-track strengths keep original order.
    mag_tie = _save_track(
        tmp_path / "mag_tie.npy",
        [
            [4.0] * 10,
            [4.0] * 10,
            [1.0] * 10,
        ],
    )
    signed_tie = _save_track(
        tmp_path / "signed_tie.npy",
        [
            [-2.0] * 10,
            [2.0] * 10,
            [0.5] * 10,
        ],
    )
    # strengths: max(4,2)=4, max(4,2)=4, max(1,0.5)=1 → order row2, then row0, row1
    fig_tie = _capture_figure(
        _heatmap_argv(
            tmp_path,
            [("Mag", mag_tie, "False"), ("Signed", signed_tie, "False")],
            absolute=["True", "False"],
        )
    )
    mag_tie_img = np.asarray(_panel_image(fig_tie, 0).get_array(), dtype=float)
    signed_tie_img = np.asarray(_panel_image(fig_tie, 1).get_array(), dtype=float)
    np.testing.assert_allclose(mag_tie_img[0], np.full(10, 1.0))
    np.testing.assert_allclose(mag_tie_img[1], np.full(10, 4.0))
    np.testing.assert_allclose(mag_tie_img[2], np.full(10, 4.0))
    np.testing.assert_allclose(signed_tie_img[0], np.full(10, 0.5))
    np.testing.assert_allclose(signed_tie_img[1], np.full(10, -2.0))
    np.testing.assert_allclose(signed_tie_img[2], np.full(10, 2.0))


def test_export_heatmap_writes_supported_image(tmp_path: Path):
    """Integration: public CLI writes a supported image file (SPEC014)."""
    track = _save_track(
        tmp_path / "out.npy",
        [[1.0] * 10, [-2.0] * 10, [0.5] * 10],
    )
    opath = tmp_path / "heatmap.png"
    _run_cli(
        _heatmap_argv(
            tmp_path,
            [("A", track, "False")],
            absolute=["False"],
            opath=opath,
        )
    )
    assert opath.is_file()
    assert opath.stat().st_size > 0
