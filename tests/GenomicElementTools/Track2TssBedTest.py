import argparse
import os
import shutil
import unittest

import numpy as np

from RGTools.BedTable import BedRegion, BedTable3

from GenomicElementTools.count_bw import CountSingleBw
from GenomicElementTools.track2tss_bed import Track2TssBed

from tests._paths import (
    BW_MN,
    BW_PL,
    FIXTURES_DIR,
    HAS_PAIRED_BW,
    SKIP_NO_PAIRED_BW,
)


class Track2TssBedTest(unittest.TestCase):
    def setUp(self):
        self._test_path = "Track2TssBedTest_temp_data"

        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)

        self._bed3_path = str(FIXTURES_DIR / "three_genes.bed3")
        self._bed6_path = str(FIXTURES_DIR / "three_genes.bed6")
        self._bed6gene_path = str(FIXTURES_DIR / "three_genes.bed6gene")

        self._pl_bw_path = str(BW_PL)
        self._mn_bw_path = str(BW_MN)

        self._pl_track_path = os.path.join(self._test_path, "procap_sig.pl.npy")
        self._mn_track_path = os.path.join(self._test_path, "procap_sig.mn.npy")

        return super().setUp()

    def tearDown(self):
        if os.path.exists(self._test_path):
            shutil.rmtree(self._test_path)

        return super().tearDown()

    def _prepare_tracks(self):
        if not HAS_PAIRED_BW:
            self.skipTest(SKIP_NO_PAIRED_BW)

        count_bw_args = argparse.Namespace()
        count_bw_args.subcommand = "count_single_bw"
        count_bw_args.bw_path = self._pl_bw_path
        count_bw_args.region_file_path = self._bed3_path
        count_bw_args.region_file_type = "bed3"
        count_bw_args.override_strand = None
        count_bw_args.quantification_type = "full_track"
        count_bw_args.opath = self._pl_track_path

        CountSingleBw.main(count_bw_args)

        count_bw_args.bw_path = self._mn_bw_path
        count_bw_args.opath = self._mn_track_path

        CountSingleBw.main(count_bw_args)

    def test_get_output_site_coord(self):
        track = np.zeros((1000,))
        track[50] = 1
        region = BedRegion("chr1", 1000, 2000)
        output = Track2TssBed.get_output_site_coord(region, track, "MaxAbsSig")
        self.assertEqual(output, 1050)

    def test_track2tss_bed(self):
        self._prepare_tracks()

        args = argparse.Namespace()
        args.subcommand = "track2tss_bed"
        args.region_file_path = self._bed3_path
        args.region_file_type = "bed3"
        args.track = self._pl_track_path
        args.opath = os.path.join(self._test_path, "output.bed")
        args.output_site = "MaxAbsSig"

        Track2TssBed.main(args)

        output_bt = BedTable3()
        output_bt.load_from_file(args.opath)
        output_df = output_bt.to_dataframe()

        self.assertEqual(len(output_bt), 3)
        self.assertEqual(output_df.loc[1, "start"], 45894511)

        args.track = self._mn_track_path

        Track2TssBed.main(args)

        output_bt = BedTable3()
        output_bt.load_from_file(args.opath)
        output_df = output_bt.to_dataframe()

        self.assertEqual(len(output_bt), 3)
        self.assertEqual(output_df.loc[1, "start"], 45894254)
