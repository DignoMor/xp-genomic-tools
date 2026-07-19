"""
# Regulatory Genome Tools (RGTools)

`RGTools` is a Python package designed for efficient handling of regulatory genomic data,
including BED-like regions, sequence extraction, motif analysis, and signal tracks.

## Overview

- **GenomicElements**: Main interface for working with genomic regions and reference genomes.
- **GeneralElements**: Base classes and shared logic for element collections.
- **ExogeneousSequences**: Specialized handling for sequences outside the reference genome.
- **MemeMotif**: Parser and scorer for MEME-formatted motifs.
- **BedTable**: Utilities for loading and manipulating BED files as pandas DataFrames.
- **ListFile**: Simple utility for reading and handling single-column list files.
"""

# Submodules (non-conflicting names bound for `from RGTools import BedTable`, etc.)
from . import BedTable
from . import BwTrack
from . import utils
from . import SNP_utils
from . import GTF_utils
from . import exceptions
from . import logging
from . import GeneralElements

# Top-level classes for convenience.
# Where a class shares its module's name (GenomicElements, ExogeneousSequences,
# MemeMotif, ListFile), the class is bound at package level; access the module
# via `import RGTools.GenomicElements` (or equivalent).
from .GenomicElements import GenomicElements
from .ExogeneousSequences import ExogeneousSequences
from .MemeMotif import MemeMotif
from .BedTable import BedTable3, BedTable6, BedTable6Plus, BedTable3Plus, BedTablePairEnd
from .ListFile import ListFile
from .BwTrack import SingleBwTrack, PairedBwTrack

__all__ = [
    # Submodules
    "BedTable",
    "BwTrack",
    "utils",
    "SNP_utils",
    "GTF_utils",
    "exceptions",
    "logging",
    "GeneralElements",
    # Top-level classes
    "GenomicElements",
    "ExogeneousSequences",
    "MemeMotif",
    "ListFile",
    "BedTable3",
    "BedTable6",
    "BedTable6Plus",
    "BedTable3Plus",
    "BedTablePairEnd",
    "SingleBwTrack",
    "PairedBwTrack",
]
