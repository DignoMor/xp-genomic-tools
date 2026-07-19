import argparse
import os
import shutil
import unittest

from RGTools.exceptions import InvalidBedRegionException
from RGTools.BedTable import BedTable6

from GenomicElementTools.pad_region import PadRegion

from tests._paths import FIXTURES_DIR


class PadRegionTest(unittest.TestCase):
    def setUp(self):
        self._test_path = "PadRegionTest_temp_data"

        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)

        self._bed3_path = str(FIXTURES_DIR / "three_genes.bed3")
        self._bed6_path = str(FIXTURES_DIR / "three_genes.bed6")
        self._bed6gene_path = str(FIXTURES_DIR / "three_genes.bed6gene")

        return super().setUp()

    def tearDown(self):
        if os.path.exists(self._test_path):
            shutil.rmtree(self._test_path)

        return super().tearDown()

    def get_pad_region_simple_args(self):
        args = argparse.Namespace()
        args.subcommand = "pad_region"
        args.region_file_path = self._bed6_path
        args.region_file_type = "bed6"
        args.upstream_pad = 100
        args.downstream_pad = 100
        args.ignore_strand = False
        args.method_resolving_invalid_region = "fallback"
        args.opath = os.path.join(self._test_path, "output.bed6")

        return args

    def test_pad_region(self):
        args = self.get_pad_region_simple_args()

        PadRegion.main(args)

        output_bt = BedTable6()
        output_bt.load_from_file(args.opath)

        self.assertEqual(len(output_bt), 3)

        self.assertEqual(output_bt.get_start_locs()[0], 75278225)
        self.assertEqual(output_bt.get_end_locs()[0], 75279426)

        # Test fall back
        args.method_resolving_invalid_region = "fallback"
        args.upstream_pad = -10000
        args.downstream_pad = -10000

        PadRegion.main(args)
        output_bt = BedTable6()
        output_bt.load_from_file(args.opath)
        self.assertEqual(output_bt.get_start_locs()[0], 75278325)
        self.assertEqual(output_bt.get_end_locs()[0], 75279326)

        # Test raise
        args.method_resolving_invalid_region = "raise"

        with self.assertRaises(InvalidBedRegionException):
            PadRegion.main(args)

        # test drop
        args.method_resolving_invalid_region = "drop"
        PadRegion.main(args)
        output_bt = BedTable6()
        output_bt.load_from_file(args.opath)
        self.assertEqual(len(output_bt), 0)
