"""Safe text output helpers for MotifTools."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

STDOUT_OUTPUT = "-"


def format_fasta(records: list[tuple[str, str]]) -> str:
    """Format sequence records as UTF-8 FASTA with LF line endings."""
    if not records:
        return ""
    lines: list[str] = []
    for record_id, sequence in records:
        lines.append(f">{record_id}")
        lines.append(sequence)
    return "\n".join(lines) + "\n"


def validate_output_flags(output: str, force: bool) -> None:
    if output == STDOUT_OUTPUT and force:
        raise ValueError("--output - cannot be combined with --force.")


def write_text_output(text: str, output: str, *, force: bool = False) -> None:
    """Write ``text`` to ``output`` or stdout when ``output`` is ``-``."""
    validate_output_flags(output, force)

    if output == STDOUT_OUTPUT:
        sys.stdout.write(text)
        return

    path = Path(output)
    parent = path.parent
    if not parent.exists():
        raise OSError(f"Output parent directory does not exist: {parent}")

    if path.exists() and not force:
        raise OSError(
            f"Refusing to overwrite existing file: {path} (use --force to replace it)"
        )

    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(text)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def write_meme_output(collection, output: str, *, force: bool = False) -> None:
    """Serialize a MemeMotif collection through the shared output contract."""
    if output == STDOUT_OUTPUT:
        collection.write_meme_file(sys.stdout)
        return

    path = Path(output)
    parent = path.parent
    if not parent.exists():
        raise OSError(f"Output parent directory does not exist: {parent}")

    if path.exists() and not force:
        raise OSError(
            f"Refusing to overwrite existing file: {path} (use --force to replace it)"
        )

    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            collection.write_meme_file(handle)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
