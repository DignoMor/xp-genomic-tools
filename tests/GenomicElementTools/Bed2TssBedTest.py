import argparse
import os
import shutil
import unittest

from RGTools.GenomicElements import GenomicElements
from RGTools.BedTable import BedTable6

from GenomicElementTools.bed2tss_bed import Bed2TssBed

from tests._paths import FIXTURES_DIR


class Bed2TssBedTest(unittest.TestCase):
    def setUp(self):
        self._test_path = "Bed2TssBedTest_temp_data"

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

    def get_bed2tssbed_simple_args(self):
        args = argparse.Namespace()
        args.subcommand = "bed2tssbed"
        args.region_file_path = self._bed6_path
        args.region_file_type = "bed6"
        args.opath = os.path.join(self._test_path, "bed2tssbed_output.bed6")
        args.output_site = "TSS"

        return args

    def test_bed2tssbed(self):
        args = self.get_bed2tssbed_simple_args()

        Bed2TssBed.main(args)

        output_bt = BedTable6()
        output_bt.load_from_file(args.opath)

        self.assertEqual(len(output_bt), 3)

        self.assertEqual(output_bt.get_start_locs()[0], 75279325)
        self.assertEqual(output_bt.get_end_locs()[0], 75279326)

    def test_bed2tssbed_bed6gene_io(self):
        args = self.get_bed2tssbed_simple_args()

        args.region_file_type = "bed6gene"
        args.region_file_path = self._bed6gene_path

        Bed2TssBed.main(args)

        out_bt = GenomicElements.BedTable6Gene()
        out_bt.load_from_file(args.opath)

        self.assertEqual(out_bt.get_start_locs()[0], 75279325)
        self.assertEqual(out_bt.get_end_locs()[0], 75279326)
        self.assertEqual(out_bt.get_region_extra_column("gene_symbol")[0], "gene2")
