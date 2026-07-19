# RGTools

Pip-installable library for regulatory genomic data (BED-like regions, sequences,
motifs, BigWig tracks). Part of **xp-genomic-tools**.

## Install

From this directory (`code/`):

```bash
pip install -e .
```

Then:

```python
from RGTools import BedTable3, GenomicElements, MemeMotif, ListFile, SingleBwTrack
```

## Layout

```
code/
  src/RGTools/     # library package
  tests/           # unit tests (separate from this package tree)
  pyproject.toml
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

Large-file fixtures (`hg38.fa`, ENCODE bigWigs) are optional. Point
`RGTOOLS_LARGE_FILES` at a directory containing them (same layout as legacy
`download_large_files.sh`); sequence / BigWig tests skip cleanly when missing.

Ensembl SNP tests need network access; set `RGTOOLS_SKIP_NETWORK_TESTS=1` to
skip them.

CLI packages (`GenomicElementTools`, etc.) are out of scope for this library-only
scaffold; entrypoints are not registered yet.

## Deviations from legacy (DignoMor/RGTools)

- **Packaging**: src-layout + `pyproject.toml` (legacy was a flat module tree /
  submodule). Import path remains `RGTools.*`. A thin `setup.py` shim is
  included so older pip (<22) can still do editable installs.
- **`__init__.py` exports**: Legacy listed submodule names in `__all__` without
  binding them (they were imported as `*_mod` aliases). Non-conflicting
  submodules (`BedTable`, `BwTrack`, `utils`, …) are now bound. Where a class
  shares its module name (`GenomicElements`, `ExogeneousSequences`, `MemeMotif`,
  `ListFile`), the **class** is bound at package level; import the module as
  `import RGTools.GenomicElements` if needed.
- **Dependencies**: Flexible minimum pins (`>=`) instead of exact pins; Python
  `>=3.9`. NumPy is constrained to `>=1.24,<2` (legacy used 1.24.x) to avoid
  NumPy-2 binary incompatibilities with older optional stack packages.
- **Not ported here**: legacy `doc/`, `scripts/`, CLI tooling.
