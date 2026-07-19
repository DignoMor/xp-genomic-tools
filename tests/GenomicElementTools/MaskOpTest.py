import argparse
import os
import shutil
import unittest

import numpy as np

from GenomicElementTools.mask_op import MaskOp
from RGTools.GenomicElements import GenomicElements

from tests._paths import FIXTURES_DIR


class MaskOpTest(unittest.TestCase):
    def setUp(self):
        self._test_dir = "mask_op_test_dir"
        if not os.path.exists(self._test_dir):
            os.makedirs(self._test_dir)

        self._region_file_path = str(FIXTURES_DIR / "three_genes.bed3")
        self._region_file_type = "bed3"
        self._mask1_path = os.path.join(self._test_dir, "mask1.npy")
        self._mask2_path = os.path.join(self._test_dir, "mask2.npy")
        self._mask3_path = os.path.join(self._test_dir, "mask3.npz")
        self._out_path = os.path.join(self._test_dir, "out.npy")

        np.save(self._mask1_path, np.array([True, False, True]))
        np.save(self._mask2_path, np.array([True, True, False]))
        np.savez(self._mask3_path, np.array([False, True, True]))

    def tearDown(self):
        if os.path.exists(self._test_dir):
            shutil.rmtree(self._test_dir)

    def test_intersect(self):
        args = argparse.Namespace(
            operation="intersect",
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            mask_npy=[self._mask1_path, self._mask2_path],
            opath=self._out_path,
        )
        MaskOp.main(args)

        ge = GenomicElements(
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            fasta_path=None,
        )
        ge.load_region_anno_from_npy("__mask__", self._out_path, anno_type="mask")
        out = ge.get_mask_arr("__mask__").reshape(-1,)

        np.testing.assert_array_equal(out, np.array([True, False, False]))
        self.assertEqual(out.dtype, bool)

    def test_union(self):
        args = argparse.Namespace(
            operation="union",
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            mask_npy=[self._mask1_path, self._mask3_path],
            opath=self._out_path,
        )
        MaskOp.main(args)

        ge = GenomicElements(
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            fasta_path=None,
        )
        ge.load_region_anno_from_npy("__mask__", self._out_path, anno_type="mask")
        out = ge.get_mask_arr("__mask__").reshape(-1,)

        np.testing.assert_array_equal(out, np.array([True, True, True]))
        self.assertEqual(out.dtype, bool)

    def test_opposite(self):
        args = argparse.Namespace(
            operation="opposite",
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            mask_npy=self._mask1_path,
            opath=self._out_path,
        )
        MaskOp.main(args)

        ge = GenomicElements(
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            fasta_path=None,
        )
        ge.load_region_anno_from_npy("__mask__", self._out_path, anno_type="mask")
        out = ge.get_mask_arr("__mask__").reshape(-1,)

        np.testing.assert_array_equal(out, np.array([False, True, False]))
        self.assertEqual(out.dtype, bool)

    def test_requires_at_least_two_masks(self):
        args = argparse.Namespace(
            operation="union",
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            mask_npy=[self._mask1_path],
            opath=self._out_path,
        )
        with self.assertRaises(ValueError):
            MaskOp.main(args)

    def test_shape_mismatch_raises(self):
        bad_mask_path = os.path.join(self._test_dir, "bad_mask.npy")
        np.save(bad_mask_path, np.array([True, False]))

        args = argparse.Namespace(
            operation="intersect",
            region_file_path=self._region_file_path,
            region_file_type=self._region_file_type,
            mask_npy=[self._mask1_path, bad_mask_path],
            opath=self._out_path,
        )
        with self.assertRaises(ValueError):
            MaskOp.main(args)
