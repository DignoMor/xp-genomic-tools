
import unittest
import shutil
import os

import pandas as pd
import numpy as np

from tests.BedTableTest import TestBedTable3
from RGTools.GenomicElements import GenomicElements
from RGTools.BedTable import BedTable3

from tests._paths import HAS_HG38, HG38_FA, SKIP_NO_HG38


class TestGenomicElements(unittest.TestCase):
    def setUp(self):

        self.__wdir = "GenomicElement_test"

        if not os.path.exists(self.__wdir):
            os.makedirs(self.__wdir)

        # Annotation-only tests do not open the fasta; sequence tests call _require_hg38().
        self.__hg38_genome_path = str(HG38_FA)
        self.__bed3_region_file_path = os.path.join(self.__wdir, "test.bed3")

        TestBedTable3._gen_test_bed_file(self.__bed3_region_file_path)

        return super().setUp()
    
    def tearDown(self):
        if os.path.exists(self.__wdir):
            shutil.rmtree(self.__wdir)
        return super().tearDown()

    def _init_GenomicElements(self):
        region_file_type = "bed3"
        region_file_path = self.__bed3_region_file_path
        genome_path = self.__hg38_genome_path
        return GenomicElements(region_file_path, region_file_type, genome_path)

    def _require_hg38(self):
        if not HAS_HG38:
            self.skipTest(SKIP_NO_HG38)

    def test_one_hot_encoding(self):
        encoding = GenomicElements.one_hot_encoding("ACGT")
        self.assertTrue((encoding == np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])).all())
    
    def test_export_exogenous_sequences(self):
        self._require_hg38()
        ge = self._init_GenomicElements()
        ge.export_exogenous_sequences(os.path.join(self.__wdir, "test.fa"))

        with open(os.path.join(self.__wdir, "test.fa"), "r") as handle:
            lines = handle.readlines()
            self.assertEqual(len(lines), 8)
            self.assertEqual(lines[0], ">chr1:1-5\n")
            self.assertEqual(lines[1], "NNNN\n")
            self.assertEqual(lines[2], ">chr1:8-12\n")
            self.assertEqual(lines[3], "NNNN\n")

    def test_get_num_regions(self):
        ge = self._init_GenomicElements()
        self.assertEqual(ge.get_num_regions(), 4)

    def test_get_region_seq(self):
        self._require_hg38()
        ge = self._init_GenomicElements()
        bin1_seq = ge.get_region_seq("chr2", 127048023, 127107288)
        self.assertEqual(bin1_seq[:10], "ACGTTTTCTG")

        nonexist_seq = ge.get_region_seq("chr24", 127048023, 127107288)
        self.assertIsNone(nonexist_seq)
    
    def test_get_all_region_seqs(self):
        self._require_hg38()
        region_file_path = os.path.join(self.__wdir, "get_region_seq_test.bed3")
        region_file_type = "bed3"
        region_bt = BedTable3()
        region_bt.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [123400, 124400],
            "end": [123500, 124500], 
        }))
        region_bt.write(region_file_path)

        ge = GenomicElements(region_file_path, 
                             region_file_type, 
                             self.__hg38_genome_path, 
                             )

        region_seq_list = ge.get_all_region_seqs()
        self.assertEqual(len(region_seq_list), 2)
        self.assertEqual(region_seq_list[0][:12], "GTGTAATTACAA")

    def test_get_all_region_one_hot(self):
        self._require_hg38()
        region_file_path = os.path.join(self.__wdir, "one_hot_test.bed3")
        region_file_type = "bed3"
        region_bt = BedTable3()
        region_bt.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [123400, 124400],
            "end": [123500, 124500], 
        }))
        region_bt.write(region_file_path)

        ge = GenomicElements(region_file_path, 
                             region_file_type, 
                             self.__hg38_genome_path, 
                             )

        one_hot_arr = ge.get_all_region_one_hot()
        self.assertEqual(one_hot_arr.shape, (2, 100, 4))
    
    def test_load_region_anno_from_npy(self):
        ge = self._init_GenomicElements()
        anno_name = "test_stat"
        npy_path = os.path.join(self.__wdir, "test.npy")
        anno_arr = np.array([1,2,3,4])
        np.save(npy_path, anno_arr)

        ge.load_region_anno_from_npy(anno_name, npy_path, anno_type="stat")
        self.assertEqual(ge.get_anno_dim(anno_name), 1)
        self.assertTrue((ge.get_stat_arr(anno_name).reshape(-1,) == anno_arr).all())

        anno_name = "test_track"
        anno_arr = np.ones((4, 4))
        np.save(npy_path, anno_arr)
        ge.load_region_anno_from_npy(anno_name, npy_path, anno_type="track")
        self.assertEqual(ge.get_anno_dim(anno_name), 4)
        self.assertTrue((ge.get_track_list(anno_name)[0] == np.ones(4)).all())

        anno_name = "test_array"
        anno_arr = np.ones((4, 2, 3))
        np.save(npy_path, anno_arr)
        ge.load_region_anno_from_npy(anno_name, npy_path, anno_type="array")
        self.assertEqual(ge.get_anno_type(anno_name), "array")
        self.assertTrue((ge.get_arr_anno(anno_name) == anno_arr).all())

        # Test loading from .npz file
        anno_name = "test_anno_npz"
        npz_path = os.path.join(self.__wdir, "test.npz")
        anno_arr = np.array([5, 6, 7, 8])
        np.savez(npz_path, anno_arr)
        
        ge.load_region_anno_from_npy(anno_name, npz_path, anno_type="stat")
        self.assertEqual(ge.get_anno_dim(anno_name), 1)
        self.assertTrue((ge.get_stat_arr(anno_name).reshape(-1,) == anno_arr).all())

        # Test error when .npz file contains multiple arrays
        multi_npz_path = os.path.join(self.__wdir, "test_multi.npz")
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([5, 6, 7, 8])
        np.savez(multi_npz_path, arr1=arr1, arr2=arr2)
        
        with self.assertRaises(ValueError) as context:
            ge.load_region_anno_from_npy("test_multi", multi_npz_path)
        self.assertIn("contains multiple arrays", str(context.exception))
        self.assertIn("Available keys", str(context.exception))
    
    def test_load_region_stat_from_arr_basic(self):
        ge = self._init_GenomicElements()
        anno_name = "test_anno"
        anno_arr = np.array([1,2,3,4])

        ge.load_region_stat_from_arr(anno_name, anno_arr)

        self.assertEqual(ge.get_anno_dim(anno_name), 1)
        self.assertTrue((ge.get_stat_arr(anno_name).reshape(-1,) == anno_arr).all())
        self.assertEqual(ge.get_anno_type(anno_name), "stat")

    def test_load_region_array_from_arr(self):
        ge = self._init_GenomicElements()
        anno_name = "test_array"
        anno_arr = np.random.randn(4, 3, 2)
        ge.load_region_array_from_arr(anno_name, anno_arr)
        self.assertEqual(ge.get_anno_type(anno_name), "array")
        np.testing.assert_array_equal(ge.get_arr_anno(anno_name), anno_arr)
        np.testing.assert_array_equal(ge.get_region_array_by_index(anno_name, 2), anno_arr[2])

    def test_load_region_track_from_list(self):
        # Create a bed file with different lengths
        hetero_bed = os.path.join(self.__wdir, "hetero_load_list.bed")
        with open(hetero_bed, "w") as f:
            f.write("chr1\t123400\t123500\n") # 100bp
            f.write("chr1\t124400\t124450\n") # 50bp
        
        ge = GenomicElements(hetero_bed, "bed3", self.__hg38_genome_path)
        
        # Correct lengths: 100 and 50
        anno_list = [np.ones(100), np.ones(50)]
        ge.load_region_track_from_list("test_list", anno_list)
        
        output = ge.get_track_list("test_list")
        # Max length should be 100
        self.assertEqual(len(output), 2)
        # First region should be all ones
        self.assertTrue(np.all(output[0] == 1))
        # Second region should be all ones
        self.assertTrue(np.all(output[1] == 1))

        # Test type-specific getter by index
        self.assertTrue(np.all(ge.get_region_track_by_index("test_list", 0) == np.ones(100)))
        self.assertTrue(np.all(ge.get_region_track_by_index("test_list", 1) == np.ones(50)))

    def test_load_region_stat_from_arr_shapes(self):
        ge = self._init_GenomicElements()
        # Accept (N,) and store as (N,1)
        anno = np.array([10, 20, 30, 40])
        ge.load_region_stat_from_arr("stat1", anno)
        self.assertEqual(ge.get_stat_arr("stat1").shape, (4, 1))
        self.assertEqual(ge.get_anno_type("stat1"), "stat")
        # Accept (N,1) unchanged
        anno2 = np.array([[1], [2], [3], [4]])
        ge.load_region_stat_from_arr("stat2", anno2)
        self.assertEqual(ge.get_stat_arr("stat2").shape, (4, 1))
        self.assertEqual(ge.get_anno_type("stat2"), "stat")

    def test_load_region_track_from_list_requires_region_lengths(self):
        ge = self._init_GenomicElements()
        # Track list with correct per-region lengths
        track = [np.ones(4), np.ones(4), np.ones(4), np.ones(4)]
        ge.load_region_track_from_list("track_ok", track)
        self.assertEqual(ge.get_anno_type("track_ok"), "track")
        self.assertEqual(ge.get_anno_dim("track_ok"), 4)
        # Track list with wrong per-region length should raise
        bad_track = [np.ones(3), np.ones(4), np.ones(4), np.ones(4)]
        with self.assertRaises(ValueError):
            ge.load_region_track_from_list("track_bad", bad_track)

    def test_get_region_type_specific_by_index(self):
        ge = self._init_GenomicElements()
        ge.load_region_stat_from_arr("stat", np.array([1, 2, 3, 4]))
        self.assertEqual(ge.get_region_stat_by_index("stat", 2), 3)
        ge.load_region_track_from_list("track", [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([8, 9, 10, 11]),
            np.array([12, 13, 14, 15]),
        ])
        # Regions are length-homogeneous length=4 in test data
        self.assertTrue((ge.get_region_track_by_index("track", 1) == np.array([4,5,6,7])).all())
    
    def test_get_track_list(self):
        # Create a bed file with different lengths
        hetero_bed = os.path.join(self.__wdir, "hetero_get_list.bed")
        with open(hetero_bed, "w") as f:
            f.write("chr1\t123400\t123500\n") # 100bp
            f.write("chr1\t124400\t124450\n") # 50bp
        
        ge = GenomicElements(hetero_bed, "bed3", self.__hg38_genome_path)
        
        # Test with track
        anno_list = [np.ones(100), np.zeros(50)]
        ge.load_region_track_from_list("test_track", anno_list)
        
        output_list = ge.get_track_list("test_track")
        self.assertEqual(len(output_list), 2)
        self.assertTrue(np.all(output_list[0] == 1))
        self.assertEqual(len(output_list[0]), 100)
        self.assertTrue(np.all(output_list[1] == 0))
        self.assertEqual(len(output_list[1]), 50)

    def test_save_anno_npy(self):
        ge = self._init_GenomicElements()
        anno_name = "test_anno"
        anno_arr = np.array([1,2,3,4])
        ge.load_region_stat_from_arr(anno_name, anno_arr)

        npy_path = os.path.join(self.__wdir, "test.npy")
        ge.save_anno_npy(anno_name, npy_path)

        self.assertTrue(os.path.exists(npy_path))
        self.assertTrue((np.load(npy_path).reshape(-1,) == anno_arr).all())

    def test_apply_logical_filter(self):
        ge = self._init_GenomicElements()
        anno_name = "test_anno"
        anno_arr = np.array([1,2,3,4])
        ge.load_region_stat_from_arr(anno_name, anno_arr)

        # Track aligned to max len (4)
        track_arr = [
            np.array([0, 1, 2, 3]),
            np.array([4, 5, 6, 7]),
            np.array([8, 9, 10, 11]),
            np.array([12, 13, 14, 15]),
        ]
        ge.load_region_track_from_list("track", track_arr)

        logical = np.array([True, False, True, False])

        new_region_file_path = os.path.join(self.__wdir, "new_region.bed3")
        new_ge = ge.apply_logical_filter(logical, new_region_file_path)

        self.assertEqual(new_ge.get_num_regions(), 2)
        self.assertEqual(new_ge.get_anno_dim(anno_name), 1)
        self.assertTrue((new_ge.get_stat_arr(anno_name).reshape(-1,) == np.array([1,3])).all())
        self.assertEqual(new_ge.get_anno_type(anno_name), "stat")

        # Track should be trimmed to new max len after filter (regions kept are same length=4 here)
        self.assertEqual(new_ge.get_anno_type("track"), "track")
        output_track_list = new_ge.get_track_list("track")
        self.assertEqual(len(output_track_list), 2)
        np.testing.assert_array_equal(output_track_list[0], np.array([0,1,2,3]))
        np.testing.assert_array_equal(output_track_list[1], np.array([8,9,10,11]))

    def test_apply_logical_filter_requires_boolean_dtype(self):
        ge = self._init_GenomicElements()
        with self.assertRaises(ValueError) as context:
            ge.apply_logical_filter(np.array([1, 0, 1, 0]), os.path.join(self.__wdir, "bad_filter.bed3"))
        self.assertIn("boolean dtype", str(context.exception))

    def test_load_region_stat_from_arr(self):
        ge = self._init_GenomicElements()
        # Test loading stats from a 1D numpy array
        stat_arr = np.array([10.5, 20.5, 30.5, 40.5])
        ge.load_region_stat_from_arr("stat_arr", stat_arr)
        
        self.assertEqual(ge.get_anno_type("stat_arr"), "stat")
        self.assertEqual(ge.get_stat_arr("stat_arr").shape, (4, 1))
        np.testing.assert_array_equal(ge.get_stat_arr("stat_arr").reshape(-1,), 
                                      np.array([10.5, 20.5, 30.5, 40.5]))
        
        # Test loading stats from a 2D array with shape (N, 1)
        stat_arr_2d = np.array([[1.0], [2.0], [3.0], [4.0]])
        ge.load_region_stat_from_arr("stat_arr_2d", stat_arr_2d)
        self.assertEqual(ge.get_anno_type("stat_arr_2d"), "stat")
        np.testing.assert_array_equal(ge.get_stat_arr("stat_arr_2d").reshape(-1,), 
                                      np.array([1.0, 2.0, 3.0, 4.0]))
        
        # Test error when array length doesn't match
        with self.assertRaises(ValueError):
            ge.load_region_stat_from_arr("bad_stat", np.array([1, 2, 3]))
        
        # Test error when array has wrong shape
        with self.assertRaises(ValueError) as context:
            ge.load_region_stat_from_arr("bad_stat2", np.array([[1, 2], [3, 4]]))
        self.assertIn("does not match number of regions", str(context.exception))

    def test_mask_annotation_type(self):
        ge = self._init_GenomicElements()
        # Test loading boolean array as 1D -> should be mask
        mask_arr_1d = np.array([True, False, True, False])
        ge.load_mask_from_arr("mask_1d", mask_arr_1d)
        self.assertEqual(ge.get_anno_type("mask_1d"), "mask")
        self.assertEqual(ge.get_mask_arr("mask_1d").shape, (4, 1))
        self.assertEqual(ge.get_mask_arr("mask_1d").dtype, bool)
        
        # Test type-specific index getter with mask
        self.assertEqual(ge.get_region_mask_by_index("mask_1d", 0), True)
        self.assertEqual(ge.get_region_mask_by_index("mask_1d", 1), False)

        # Test annotation_type is "mask"
        self.assertEqual(ge.get_anno_type("mask_1d"), "mask")

    def test_load_mask_from_arr(self):
        ge = self._init_GenomicElements()
        # Test loading mask from a 1D numpy array of booleans
        mask_arr = np.array([True, False, True, False])
        ge.load_mask_from_arr("mask_arr", mask_arr)
        
        self.assertEqual(ge.get_anno_type("mask_arr"), "mask")
        self.assertEqual(ge.get_mask_arr("mask_arr").shape, (4, 1))
        self.assertEqual(ge.get_mask_arr("mask_arr").dtype, bool)
        np.testing.assert_array_equal(ge.get_mask_arr("mask_arr").reshape(-1,), 
                                      np.array([True, False, True, False]))
        
        # Test loading mask from a 2D array with shape (N, 1)
        mask_arr_2d = np.array([[True], [False], [True], [False]])
        ge.load_mask_from_arr("mask_arr_2d", mask_arr_2d)
        self.assertEqual(ge.get_anno_type("mask_arr_2d"), "mask")
        np.testing.assert_array_equal(ge.get_mask_arr("mask_arr_2d").reshape(-1,), 
                                      np.array([True, False, True, False]))
        
        # Test error when array length doesn't match
        with self.assertRaises(ValueError):
            ge.load_mask_from_arr("bad_mask", np.array([True, False, True]))
        
        # Test error when array has wrong shape
        with self.assertRaises(ValueError) as context:
            ge.load_mask_from_arr("bad_mask2", np.array([[True, False], [True, False]]))
        self.assertIn("does not match number of regions", str(context.exception))
        
        # Non-boolean dtypes should be rejected explicitly.
        with self.assertRaises(ValueError) as context:
            ge.load_mask_from_arr("mask_arr_int", np.array([1, 0, 1, 0]))
        self.assertIn("boolean dtype", str(context.exception))

    def test_merge_genomic_elements(self):
        left_path = os.path.join(self.__wdir, "merge_left.bed3")
        right_path = os.path.join(self.__wdir, "merge_right.bed3")
        output_unsorted_path = os.path.join(self.__wdir, "merge_out_unsorted.bed3")
        output_sorted_path = os.path.join(self.__wdir, "merge_out_sorted.bed3")

        left_bt = BedTable3(enable_sort=False)
        left_bt.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr2", "chr2"],
            "start": [100, 200],
            "end": [110, 210],
        }))
        left_bt.write(left_path)

        right_bt = BedTable3(enable_sort=False)
        right_bt.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [50, 300],
            "end": [60, 310],
        }))
        right_bt.write(right_path)

        left_ge = GenomicElements(left_path, "bed3", self.__hg38_genome_path)
        right_ge = GenomicElements(right_path, "bed3", self.__hg38_genome_path)
        left_ge.load_region_stat_from_arr("stat", np.array([100, 200]))
        right_ge.load_region_stat_from_arr("stat", np.array([10, 20]))
        left_ge.load_region_track_from_list("track", [np.array([1] * 10), np.array([2] * 10)])
        right_ge.load_region_track_from_list("track", [np.array([3] * 10), np.array([4] * 10)])

        merged_unsorted = GenomicElements.merge_genomic_elements(
            left_ge, right_ge, output_unsorted_path, ["stat", "track"], sort_new_ge=False
        )
        self.assertEqual(merged_unsorted.get_num_regions(), 4)
        unsorted_df = merged_unsorted.get_region_bed_table().to_dataframe()
        self.assertEqual(unsorted_df.iloc[0]["chrom"], "chr2")
        self.assertEqual(unsorted_df.iloc[2]["chrom"], "chr1")
        np.testing.assert_array_equal(
            merged_unsorted.get_stat_arr("stat").reshape(-1,), np.array([100, 200, 10, 20])
        )
        self.assertEqual([x[0] for x in merged_unsorted.get_track_list("track")], [1, 2, 3, 4])

        merged_sorted = GenomicElements.merge_genomic_elements(
            left_ge, right_ge, output_sorted_path, ["stat", "track"], sort_new_ge=True
        )
        sorted_df = merged_sorted.get_region_bed_table().to_dataframe()
        self.assertEqual(sorted_df.iloc[0]["chrom"], "chr1")
        self.assertEqual(sorted_df.iloc[1]["chrom"], "chr1")
        self.assertEqual(sorted_df.iloc[2]["chrom"], "chr2")
        self.assertEqual(sorted_df.iloc[3]["chrom"], "chr2")
        np.testing.assert_array_equal(
            merged_sorted.get_stat_arr("stat").reshape(-1,), np.array([10, 20, 100, 200])
        )
        self.assertEqual([x[0] for x in merged_sorted.get_track_list("track")], [3, 4, 1, 2])

        # Different fasta paths should fail.
        right_ge_diff_fasta = GenomicElements(right_path, "bed3", "dummy.fa")
        with self.assertRaises(ValueError):
            GenomicElements.merge_genomic_elements(
                left_ge, right_ge_diff_fasta, os.path.join(self.__wdir, "bad_fasta.bed3"), []
            )

        # Different region file types should fail.
        bed6_path = os.path.join(self.__wdir, "merge_other.bed6")
        with open(bed6_path, "w") as f:
            f.write("chr1\t10\t20\tname1\t10\t+\n")
        ge_bed6 = GenomicElements(bed6_path, "bed6", self.__hg38_genome_path)
        with self.assertRaises(ValueError):
            GenomicElements.merge_genomic_elements(
                left_ge, ge_bed6, os.path.join(self.__wdir, "bad_type.bed3"), []
            )

