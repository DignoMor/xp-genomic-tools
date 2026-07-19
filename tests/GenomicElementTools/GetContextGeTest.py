import argparse
import os
import shutil
import unittest

import pandas as pd
import numpy as np

from RGTools.BedTable import BedTable3

from GenomicElementTools.get_context_ge import GetContextGe


class GetContextGeTest(unittest.TestCase):
    def setUp(self):
        self._test_path = "GetContextGeTest_temp_data"
        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)
        return super().setUp()

    def tearDown(self):
        if os.path.exists(self._test_path):
            shutil.rmtree(self._test_path)
        return super().tearDown()

    def _write_bed3(self, path, rows):
        bt = BedTable3(enable_sort=False)
        bt.load_from_dataframe(
            pd.DataFrame(rows, columns=["chrom", "start", "end"])
        )
        bt.write(path)

    def _get_base_args(self):
        args = argparse.Namespace()
        args.subcommand = "get_context_ge"
        args.method = "nearest"
        args.region_file_type = "bed3"
        args.context_file_type = "bed3"
        args.region_file_path = os.path.join(self._test_path, "regions.bed3")
        args.context_file_path = os.path.join(self._test_path, "context.bed3")
        args.opath = os.path.join(self._test_path, "output.bed3")
        return args

    def test_nearest_selects_expected_context(self):
        args = self._get_base_args()
        self._write_bed3(
            args.region_file_path,
            [
                ("chr1", 100, 120),
                ("chr1", 300, 320),
                ("chr2", 50, 70),
            ],
        )
        # chr2 contexts intentionally tie at distance 10; first one should be selected.
        self._write_bed3(
            args.context_file_path,
            [
                ("chr1", 10, 20),
                ("chr1", 130, 140),
                ("chr1", 260, 280),
                ("chr1", 500, 520),
                ("chr2", 10, 40),
                ("chr2", 80, 90),
            ],
        )

        GetContextGe.main(args)

        output_bt = BedTable3()
        output_bt.load_from_file(args.opath)
        output_df = output_bt.to_dataframe()

        self.assertEqual(len(output_df), 3)
        self.assertEqual(
            list(zip(output_df["chrom"], output_df["start"], output_df["end"])),
            [
                ("chr1", 130, 140),
                ("chr1", 260, 280),
                ("chr2", 10, 40),
            ],
        )

    def test_nearest_raises_when_chrom_has_no_context(self):
        args = self._get_base_args()
        self._write_bed3(args.region_file_path, [("chr3", 100, 120)])
        self._write_bed3(args.context_file_path, [("chr1", 10, 20)])

        with self.assertRaises(ValueError) as context:
            GetContextGe.main(args)

        self.assertIn("No context regions found on chromosome 'chr3'", str(context.exception))

    def test_windowed_argmax_selects_max_stat_context_in_each_window(self):
        args = self._get_base_args()
        args.method = "windowed_argmax"
        args.context_stat_path = os.path.join(self._test_path, "context_stat.npy")

        # region_file_path here stores windows (e.g., padded input elements).
        self._write_bed3(
            args.region_file_path,
            [
                ("chr1", 0, 200),
                ("chr1", 200, 400),
                ("chr2", 0, 120),
            ],
        )
        self._write_bed3(
            args.context_file_path,
            [
                ("chr1", 10, 20),    # in window1, stat=1
                ("chr1", 130, 140),  # in window1, stat=4 (selected)
                ("chr1", 220, 230),  # in window2, stat=5 (selected, tie by order)
                ("chr1", 260, 270),  # in window2, stat=5
                ("chr2", 80, 90),    # in window3, stat=3 (selected)
            ],
        )
        np.save(args.context_stat_path, np.array([1.0, 4.0, 5.0, 5.0, 3.0]))

        GetContextGe.main(args)

        output_bt = BedTable3()
        output_bt.load_from_file(args.opath)
        output_df = output_bt.to_dataframe()

        self.assertEqual(len(output_df), 3)
        self.assertEqual(
            list(zip(output_df["chrom"], output_df["start"], output_df["end"])),
            [
                ("chr1", 130, 140),
                ("chr1", 220, 230),
                ("chr2", 80, 90),
            ],
        )
