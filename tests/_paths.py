"""Shared path helpers for RGTools tests."""

from __future__ import annotations

import os
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
CODE_ROOT = TESTS_DIR.parent

# Optional large fixtures (hg38.fa, bigWigs). Override with RGTOOLS_LARGE_FILES.
_DEFAULT_LARGE_FILES_CANDIDATES = (
    CODE_ROOT / "large_files",
    CODE_ROOT.parent / "sandbox" / "RGTools-legacy" / "large_files",
    Path.home() / "RGTools" / "large_files",
)


def large_files_dir() -> Path:
    env = os.environ.get("RGTOOLS_LARGE_FILES")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in _DEFAULT_LARGE_FILES_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    return _DEFAULT_LARGE_FILES_CANDIDATES[0]


def large_file(*parts: str) -> Path:
    return large_files_dir().joinpath(*parts)


def has_file(path: Path) -> bool:
    return path.is_file()


HG38_FA = large_file("hg38.fa")
BW_PL = large_file("ENCFF565BWR.pl.bw")
BW_MN = large_file("ENCFF775FNU.mn.bw")

HAS_HG38 = has_file(HG38_FA)
HAS_BW_PL = has_file(BW_PL)
HAS_BW_MN = has_file(BW_MN)
HAS_PAIRED_BW = HAS_BW_PL and HAS_BW_MN

SAMPLE_MEME = FIXTURES_DIR / "sample-dna-motif.meme"

SKIP_NO_HG38 = "Requires hg38.fa; set RGTOOLS_LARGE_FILES to the large_files directory"
SKIP_NO_BW_PL = "Requires ENCFF565BWR.pl.bw; set RGTOOLS_LARGE_FILES"
SKIP_NO_PAIRED_BW = "Requires paired bigWig files; set RGTOOLS_LARGE_FILES"
