# RGTools / xp-genomic-tools

Pip-installable library (`RGTools`) and CLIs (`GenomicElementTools`,
`ExogenousSequenceTools`) for regulatory genomic data (BED-like regions,
sequences, motifs, BigWig tracks). Part of **xp-genomic-tools**.

## Install

From this directory (`code/`):

```bash
pip install -e ".[dev]"
```

Then:

```python
from RGTools import BedTable3, GenomicElements, MemeMotif, ListFile, SingleBwTrack
from GenomicElementTools.cli import GenomicElementTools
from ExogenousSequenceTools.cli import ExogenousSequenceTools
```

## CLI

After install, the console scripts are on `PATH`:

### GenomicElementTools

```bash
GenomicElementTools --help
GenomicElementTools pad_region --help
```

You can also run the package module:

```bash
python -m GenomicElementTools --help
```

Subcommands include `count_single_bw`, `count_paired_bw`, `pad_region`,
`bed2tssbed`, `onehot`, `motif_search`, `track2tss_bed`, `filter_motif_score`,
`export`, `import`, `get_context_ge`, `mask_op`, and
`select_tss_relative_track`.

### ExogenousSequenceTools

```bash
ExogenousSequenceTools --help
ExogenousSequenceTools assemble --help
```

```bash
python -m ExogenousSequenceTools --help
```

Subcommands include `assemble`, `track_dim_reduction`, `mutagenesis`,
`gen_track`, `print_stat`, `motif_search`, and `onehot`.

## Layout

```
code/
  src/RGTools/                   # library package
  src/GenomicElementTools/       # GenomicElementTools CLI package
  src/ExogenousSequenceTools/   # ExogenousSequenceTools CLI package
  tests/                         # unit tests (separate from this package tree)
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

## Deviations from legacy (DignoMor/RGTools)

- **Packaging**: src-layout + `pyproject.toml` (legacy was a flat module tree /
  submodule). Import path remains `RGTools.*`. A thin `setup.py` shim is
  included so older pip (<22) can still do editable installs.
- **`__init__.py` exports**: Legacy listed submodule names in `__all__` without
  binding them (they were imported as `*_mod` aliases). Non-conflicting
  submodules (`BedTable`, `BwTrack`, `utils`, …) are now bound. Where a class
  shares its module name (`GenomicElements`, `ExogenousSequences`, `MemeMotif`,
  `ListFile`), the **class** is bound at package level; import the module as
  `import RGTools.GenomicElements` if needed.
- **Dependencies**: Flexible minimum pins (`>=`) instead of exact pins; Python
  `>=3.9`. NumPy is constrained to `>=1.24,<2` (legacy used 1.24.x) to avoid
  NumPy-2 binary incompatibilities with older optional stack packages.
  `matplotlib` is a declared dependency (needed by `GenomicElementTools export Heatmap`).
- **CLI packaging**: Legacy `GenomicElementTool` / `ExogeneousSequenceTool`
  (flat scripts + RGTools submodule) are packaged as `GenomicElementTools` /
  `ExogenousSequenceTools` under `src/` with relative imports and console-script
  entrypoints. `CountTableTools` is not registered yet.
- **Not ported here**: legacy `doc/` and `scripts/` from RGTools.
