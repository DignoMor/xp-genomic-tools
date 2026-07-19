import argparse
import os
import shutil
import unittest

import numpy as np

from GenomicElementTools.motif_search import MotifSearch
from GenomicElementTools.filter_motif_score import FilterMotifScore
from RGTools.GenomicElements import GenomicElements

from tests._paths import (
    FIXTURES_DIR,
    HAS_HG38,
    HG38_FA,
    SAMPLE_MEME,
    SKIP_NO_HG38,
)


@unittest.skipUnless(HAS_HG38, SKIP_NO_HG38)
class FilterMotifScoreTest(unittest.TestCase):
    def setUp(self):
        self._test_path = "filter_motif_score_test_dir"

        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)

        self._hg38_fasta_path = str(HG38_FA)
        self._bed6_path = str(FIXTURES_DIR / "three_genes.bed6")
        self._meme_motif_path = str(SAMPLE_MEME)

        args = argparse.Namespace()
        args.subcommand = "motif_search"
        args.fasta_path = self._hg38_fasta_path
        args.region_file_path = self._bed6_path
        args.region_file_type = "bed6"
        args.motif_file = self._meme_motif_path
        args.output_header = os.path.join(self._test_path, "three_genes.motif_search")
        args.estimate_background_freq = True
        args.strand = "-"
        MotifSearch.main(args)

    def tearDown(self):
        if os.path.exists(self._test_path):
            shutil.rmtree(self._test_path)
        super().tearDown()

    def get_filter_motif_score_simple_args(self):
        args = argparse.Namespace()

        args.subcommand = "filter_motif_score"
        args.region_file_path = self._bed6_path
        args.region_file_type = "bed6"
        args.motif_search_npy = os.path.join(
            self._test_path, "three_genes.motif_search.crp.npy"
        )
        args.output_header = os.path.join(self._test_path, "three_genes.crp.filtered")
        args.filter_base = 649
        args.min_score = 0.0
        args.max_score = np.inf

        return args

    def test_filter_motif_score(self):
        args = self.get_filter_motif_score_simple_args()
        FilterMotifScore.main(args)

        filtered_ge = GenomicElements(
            region_file_path=args.output_header + ".bed",
            region_file_type=args.region_file_type,
            fasta_path=None,
        )

        filtered_ge.load_region_anno_from_npy(
            "motif",
            args.output_header + ".motif.npy",
            anno_type="track",
        )

        self.assertEqual(len(filtered_ge.get_track_list("motif")), 1)


if __name__ == "__main__":
    unittest.main()
