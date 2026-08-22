"""Protected paired NPY/NPZ publication for coupled annotation artifacts.

Remnant name patterns (beside each destination):

- Staging: ``.<stem>.staging<suffix>``  (e.g. ``.coord.staging.npy``,
  ``.mask.staging.npz``) so the staged file keeps a supported annotation suffix.
- Backup:  ``.<basename>.bak``          (e.g. ``.coord.npy.bak``)

Interrupted staging or backup remnants are detected and reported; this module
never deletes them automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

SUPPORTED_ANNO_SUFFIXES = (".npy", ".npz")


def annotation_suffix(path: Path | str) -> str:
    """Return a lowercase annotation suffix or raise for unsupported paths."""
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_ANNO_SUFFIXES:
        raise ValueError(
            f"Unsupported annotation suffix {Path(path).suffix!r} for {path}; "
            f"use {', '.join(SUPPORTED_ANNO_SUFFIXES)}."
        )
    return suffix


def staging_path(destination: Path) -> Path:
    """Stable staging remnant: ``.<stem>.staging<suffix>`` beside destination."""
    return destination.parent / f".{destination.stem}.staging{destination.suffix}"


def backup_path(destination: Path) -> Path:
    """Stable backup remnant path: ``.<basename>.bak`` beside destination."""
    return destination.parent / f".{destination.name}.bak"


def save_annotation_array(path: Path | str, array: np.ndarray) -> None:
    """Write one array as ``.npy`` or single-array ``.npz`` from the path suffix."""
    path = Path(path)
    suffix = annotation_suffix(path)
    if suffix == ".npy":
        np.save(path, array)
    else:
        np.savez_compressed(path, array)


def _existing_remnants(destinations: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for destination in destinations:
        for remnant in (staging_path(destination), backup_path(destination)):
            if remnant.exists():
                found.append(remnant)
    return found


def publish_paired_annotations(
    *,
    coordinate_path: Path | str,
    coordinate_array: np.ndarray,
    mask_path: Path | str,
    mask_array: np.ndarray,
    force: bool = False,
) -> None:
    """Publish coordinate and mask arrays with protected paired replacement.

    Both arrays must already be fully computed and validated. Destinations are
    not created or changed until staging succeeds. Existing destinations require
    ``force``. On an ordinary failure after backups begin, destinations are
    restored from backups when available.
    """
    coord_dest = Path(coordinate_path)
    mask_dest = Path(mask_path)
    annotation_suffix(coord_dest)
    annotation_suffix(mask_dest)

    for dest in (coord_dest, mask_dest):
        parent = dest.parent
        if not parent.exists():
            raise OSError(f"Output parent directory does not exist: {parent}")

    remnants = _existing_remnants((coord_dest, mask_dest))
    if remnants:
        names = ", ".join(str(path) for path in remnants)
        raise OSError(
            "Interrupted publication remnants detected beside destinations; "
            f"resolve manually before retrying: {names}"
        )

    existing = [dest for dest in (coord_dest, mask_dest) if dest.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise OSError(
            f"Refusing to overwrite existing file(s): {names} "
            "(use --force to replace them)"
        )

    coord_staging = staging_path(coord_dest)
    mask_staging = staging_path(mask_dest)
    coord_backup = backup_path(coord_dest)
    mask_backup = backup_path(mask_dest)

    save_annotation_array(coord_staging, coordinate_array)
    save_annotation_array(mask_staging, mask_array)

    backed_up: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        if coord_dest.exists():
            os.replace(coord_dest, coord_backup)
            backed_up.append((coord_dest, coord_backup))
        if mask_dest.exists():
            os.replace(mask_dest, mask_backup)
            backed_up.append((mask_dest, mask_backup))

        os.replace(coord_staging, coord_dest)
        published.append(coord_dest)
        os.replace(mask_staging, mask_dest)
        published.append(mask_dest)
    except Exception:
        for dest, backup in backed_up:
            if backup.exists():
                os.replace(backup, dest)
        for dest in published:
            if dest.exists() and not any(dest == pair[0] for pair in backed_up):
                try:
                    dest.unlink()
                except OSError:
                    pass
        for staging in (coord_staging, mask_staging):
            if staging.exists():
                try:
                    staging.unlink()
                except OSError:
                    pass
        raise

    for _dest, backup in backed_up:
        if backup.exists():
            backup.unlink()
