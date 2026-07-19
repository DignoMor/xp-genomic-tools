import argparse
import os
import shutil
import unittest

import numpy as np

from GenomicElementTools.motif_search import MotifSearch
from RGTools.GenomicElements import GenomicElements

from tests._paths import (
    FIXTURES_DIR,
    HAS_HG38,
    HG38_FA,
    SAMPLE_MEME,
    SKIP_NO_HG38,
)


@unittest.skipUnless(HAS_HG38, SKIP_NO_HG38)
class MotifSearchTest(unittest.TestCase):
    def setUp(self):
        self._test_path = "motif_search_test_dir"

        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)

        self._hg38_fasta_path = str(HG38_FA)
        self._bed6_path = str(FIXTURES_DIR / "three_genes.bed6")
        self._meme_motif_path = str(SAMPLE_MEME)

    def tearDown(self):
        if os.path.exists(self._test_path):
            shutil.rmtree(self._test_path)
        super().tearDown()

    def get_motif_search_simple_args(self):
        args = argparse.Namespace()

        args.subcommand = "motif_search"
        args.fasta_path = self._hg38_fasta_path
        args.region_file_path = self._bed6_path
        args.region_file_type = "bed6"
        args.motif_file = self._meme_motif_path
        args.output_header = os.path.join(self._test_path, "three_genes.motif_search")
        args.estimate_background_freq = True
        args.strand = "+"

        return args

    def test_main(self):
        args = self.get_motif_search_simple_args()

        MotifSearch.main(args)

        output_ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=args.fasta_path,
        )

        output_ge.load_region_anno_from_npy(
            "CRP",
            os.path.join(args.output_header + ".crp.npy"),
            anno_type="track",
        )

        output_ge.load_region_anno_from_npy(
            "LexA",
            os.path.join(args.output_header + ".lexA.npy"),
            anno_type="track",
        )

        crp_track_list = output_ge.get_track_list("CRP")
        lexA_track_list = output_ge.get_track_list("LexA")

        self.assertEqual(len(crp_track_list), 3)
        self.assertEqual(len(lexA_track_list[0]), 1001)

        self.assertFalse(all([np.all(t == 0) for t in crp_track_list]))
        self.assertFalse(all([np.all(np.isinf(t)) for t in crp_track_list]))

    def test_main_reverse_strand(self):
        args = self.get_motif_search_simple_args()
        args.strand = "-"

        MotifSearch.main(args)

        output_ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=args.fasta_path,
        )

        output_ge.load_region_anno_from_npy(
            "CRP",
            os.path.join(args.output_header + ".crp.npy"),
            anno_type="track",
        )

        crp_track_list = output_ge.get_track_list("CRP")

        self.assertEqual(len(crp_track_list), 3)
        self.assertEqual(len(crp_track_list[0]), 1001)

        self.assertFalse(all([np.all(t == 0) for t in crp_track_list]))
        self.assertFalse(all([np.all(np.isinf(t)) for t in crp_track_list]))

    def test_main_both_strands(self):
        args = self.get_motif_search_simple_args()
        args.strand = "both"

        MotifSearch.main(args)

        output_ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=args.fasta_path,
        )

        output_ge.load_region_anno_from_npy(
            "CRP",
            os.path.join(args.output_header + ".crp.npy"),
            anno_type="track",
        )

        crp_track_list = output_ge.get_track_list("CRP")

        self.assertEqual(len(crp_track_list), 3)
        self.assertEqual(len(crp_track_list[0]), 1001)

        self.assertFalse(all([np.all(t == 0) for t in crp_track_list]))
        self.assertFalse(all([np.all(np.isinf(t)) for t in crp_track_list]))

        args_fwd = self.get_motif_search_simple_args()
        args_fwd.strand = "+"
        MotifSearch.main(args_fwd)
        output_ge_fwd = GenomicElements(
            region_file_path=args_fwd.region_file_path,
            region_file_type=args_fwd.region_file_type,
            fasta_path=args_fwd.fasta_path,
        )
        output_ge_fwd.load_region_anno_from_npy(
            "CRP",
            os.path.join(args_fwd.output_header + ".crp.npy"),
            anno_type="track",
        )
        crp_track_fwd_list = output_ge_fwd.get_track_list("CRP")

        for t_both, t_fwd in zip(crp_track_list, crp_track_fwd_list):
            self.assertTrue(np.all(t_both >= t_fwd))


if __name__ == "__main__":
    unittest.main()
