"""Fail if Exogeneous/exogeneous appears outside the Delivery Spec allowlist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
SPELLING_RE = re.compile(r"Exogeneous|exogeneous")

SCAN_ROOTS = (
    CODE_ROOT / "src",
    CODE_ROOT / "tests",
    CODE_ROOT / "pyproject.toml",
    CODE_ROOT / "README.md",
)

SKIP_DIR_NAMES = {".venv", "__pycache__", ".pytest_cache", "build", "dist"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".whl"}
SKIP_PATH_MARKERS = (".egg-info",)

ALLOWLIST_FILES = {
    CODE_ROOT / "tests" / "test_exogenous_identifier_cutover.py",
    CODE_ROOT / "tests" / "test_active_spelling_policy.py",
}

README_LEGACY_PROVENANCE_RE = re.compile(
    r"Legacy\s+`GenomicElementTool`\s*/\s*`ExogeneousSequenceTool`"
)


def _iter_scan_paths() -> list[Path]:
    paths: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            paths.append(root)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if any(marker in path.as_posix() for marker in SKIP_PATH_MARKERS):
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            paths.append(path)
    return paths


def _line_allowlisted(path: Path, line: str) -> bool:
    if path in ALLOWLIST_FILES:
        return True
    if path == CODE_ROOT / "README.md" and README_LEGACY_PROVENANCE_RE.search(line):
        return True
    return False


def _collect_violations() -> list[str]:
    violations: list[str] = []
    for path in _iter_scan_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not SPELLING_RE.search(line):
                continue
            if _line_allowlisted(path, line):
                continue
            rel = path.relative_to(CODE_ROOT)
            violations.append(f"{rel}:{line_no}: {line.strip()}")
    return violations


def test_active_surfaces_use_canonical_exogenous_spelling() -> None:
    violations = _collect_violations()
    assert not violations, "Forbidden misspellings outside allowlist:\n" + "\n".join(
        violations
    )
