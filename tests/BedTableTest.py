
import unittest
import shutil
import os

import numpy as np
import pandas as pd

from RGTools.BedTable import BedRegion, \
                       BedTable3, \
                       BedTable6, \
                       BedTable6Plus, \
                       BedTablePairEnd, \
                       BedTable3Plus

from RGTools.exceptions import InvalidBedRegionException, \
                         InvalidStrandnessException, \
                         BedTableLoadException

class TestBedRegion(unittest.TestCase):
    def test_pad_region(self):
        region = BedRegion("chr1", 12, 15, name=".", score=".", strand="-")

        padded_region = region.pad_region(1, 1)

        self.assertEqual(padded_region["chrom"], "chr1")
        self.assertEqual(padded_region["start"], 11)
        self.assertEqual(padded_region["end"], 16)

        padded_region = region.pad_region(2, -1)

        self.assertEqual(padded_region["chrom"], "chr1")
        self.assertEqual(padded_region["start"], 13)
        self.assertEqual(padded_region["end"], 17)

        padded_region = region.pad_region(2, 1, ignore_strand=True)

        self.assertEqual(padded_region["chrom"], "chr1")
        self.assertEqual(padded_region["start"], 10)
        self.assertEqual(padded_region["end"], 16)
        self.assertEqual(padded_region["strand"], "-")

        # Test invalid strand
        region = BedRegion("chr1", 12, 15, name=".", score=".", strand=".")

        with self.assertRaises(InvalidStrandnessException):
            region.pad_region(1, 1)

        region = BedRegion("chr1", 12, 15, )

        with self.assertRaises(InvalidStrandnessException):
            region.pad_region(1, 1)
        
        # Test invalid padding

        region = BedRegion("chr1", 12, 15, name=".", score=".", strand="+")

        with self.assertRaises(InvalidBedRegionException):
            region.pad_region(-10, -1)
        
        with self.assertRaises(InvalidBedRegionException):
            region.pad_region(20, 1)


class TestBedTable(unittest.TestCase):
    # Public methods for testing BedTable
    def assertArrayEqual(self, arr1, arr2):
        '''
        equal assertion to compare two numpy arrays.
        '''
        if arr1.shape != arr2.shape:
            return self.assertFalse(True, 
                                    msg="Shape mismatch in array comparison: {} != {}".format(arr1.shape, arr2.shape),
                                    )

        return self.assertTrue((arr1 == arr2).all(), 
                               msg="Array mismatch in array comparison: {} != {}".format(arr1, arr2),
                               )

    def mutate_test_helper(self, bedtable: BedTable3) -> bool:
        '''
        Helper function to test the immutability of the class.
        Return True if the passed object is not mutated.
        '''
        return self.assertArrayEqual(self.data_df.values, 
                                     bedtable.to_dataframe().values, 
                                     )
    
    @staticmethod
    def _brute_force_search_region(bed_table: BedTable3,
                                  chrom: str,
                                  start: int,
                                  end: int,
                                  overlapping_base: int = 1,
                                  ) -> np.array:
        '''
        Reference overlap search: scan all regions (no binary search).
        '''
        result_inds = []
        for i in range(len(bed_table)):
            region = bed_table.get_region_by_index(i)
            if region["chrom"] != chrom:
                continue
            if region["end"] - start >= overlapping_base and end - region["start"] >= overlapping_base:
                result_inds.append(i)
        return np.array(result_inds, dtype=int)
    
    def setUp(self) -> None:
        '''
        Set up the test directory and data file path.
        '''
        super().setUp()
        self.data_dir = "bedTableTest_temp_data"
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.data_file = os.path.join(self.data_dir, "test.bed")

    def tearDown(self) -> None:
        '''
        Clean up test data directory.
        '''
        super().tearDown()
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

class TestBedTable3(TestBedTable):
    def setUp(self) -> None:
        super().setUp()
        self.data_df = self.__class__._gen_test_bed_file(self.data_file)

    def tearDown(self) -> None:
        super().tearDown()

    @staticmethod
    def _gen_test_bed_file(bed_path):
        '''
        Generate a test bed file.
        Return the data in pd.DataFrame format

        Keyword Arguments:
        - bed_path: str, path to the bed file.

        Return:
        - data_df: pd.DataFrame for data contained in the generated bed.
        '''
        data_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1, 8, 3, 4],
            "end": [5, 12, 7, 8],
        })

        data_df.to_csv(bed_path, 
                       sep="\t", 
                       header=False, 
                       index=False, 
                       )
        
        return data_df

    
    def test_load_from_file(self):
        bed_table = BedTable3()
        bed_table.load_from_file(self.data_file)

        self.assertArrayEqual(bed_table.to_dataframe().values, 
                              self.data_df.values,
                              )
        
    def test_load_from_dataframe(self):
        bed_table = BedTable3()
        bed_table.load_from_dataframe(self.data_df.copy())

        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

    def test_load_from_bed_regions(self):
        src_bed_table = self.__init_test_bed_table()
        regions = list(src_bed_table.iter_regions())

        bed_table = BedTable3()
        bed_table.load_from_bed_regions(regions)

        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

    def test_apply_logical_filter(self):
        logical_array = [True, False, True, False]
        logical_array = np.array(logical_array)

        bed_table = self.__init_test_bed_table()

        filtered_bed_table = bed_table.apply_logical_filter(logical_array)

        self.assertArrayEqual(filtered_bed_table.to_dataframe().values, self.data_df.loc[logical_array].values)

    def test_apply_logical_filter_requires_boolean_dtype(self):
        bed_table = self.__init_test_bed_table()
        with self.assertRaises(ValueError) as context:
            bed_table.apply_logical_filter(np.array([1, 0, 1, 0]))
        self.assertIn("boolean dtype", str(context.exception))

    def test_region_subset(self):
        bed_table = self.__init_test_bed_table()

        chrom = "chr2"
        lboundary = 2
        rboundary = 7

        subset_bed_table = bed_table.region_subset(chrom=chrom,
                                                   start=lboundary, 
                                                   end=rboundary,
                                                   )

        self.assertArrayEqual(subset_bed_table.to_dataframe().values,
                              self.data_df.loc[(self.data_df["chrom"] == chrom) & \
                                               (self.data_df["start"] >= lboundary) & \
                                               (self.data_df["end"] <= rboundary)].values,
                              )

    def test_to_dataframe(self):
        bed_table = self.__init_test_bed_table()

        exported_dataframe = bed_table.to_dataframe()

        self.assertEqual(type(exported_dataframe), type(pd.DataFrame()))
        self.assertArrayEqual(exported_dataframe.values,
                              self.data_df.values,
                              )
        
        # Test immutability
        exported_dataframe.loc[0, 0] = "chr3"
        self.mutate_test_helper(bed_table)

    def test_write(self):
        bed_table = self.__init_test_bed_table()

        write_path = os.path.join(self.data_dir, "write_test.bed")
        bed_table.write(write_path)

        new_bed_table = BedTable3()
        new_bed_table.load_from_file(write_path)

        self.assertArrayEqual(new_bed_table.to_dataframe().values,
                              bed_table.to_dataframe().values,
                              )

    def test_get_chrom_names(self):
        bed_table = self.__init_test_bed_table()

        chrom_names = bed_table.get_chrom_names()

        self.assertArrayEqual(chrom_names, self.data_df["chrom"].values)

    def test_get_start_locs(self):
        bed_table = self.__init_test_bed_table()

        start_locs = bed_table.get_start_locs()

        self.assertArrayEqual(start_locs, self.data_df["start"].values)

    def test_get_end_locs(self):
        bed_table = self.__init_test_bed_table()

        end_locs = bed_table.get_end_locs()

        self.assertArrayEqual(end_locs, self.data_df["end"].values)
    
    def test_get_region_by_index(self):
        bed_table = self.__init_test_bed_table()

        region = bed_table.get_region_by_index(0)

        self.assertEqual(region["chrom"], self.data_df.loc[0, "chrom"])
        self.assertEqual(region["start"], self.data_df.loc[0, "start"])
        self.assertEqual(region["end"], self.data_df.loc[0, "end"])

    def test_iter_regions(self):
        bed_table = self.__init_test_bed_table()
        start_locs = np.array([r["start"] for r in bed_table.iter_regions()])

        self.assertArrayEqual(start_locs, self.data_df["start"].values)
    
    def test_concat(self):
        bed_table = self.__init_test_bed_table()

        new_bed_table = BedTable3()
        new_bed_table.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [2, 80, 50],
            "end": [6, 120, 70],}), 
            )

        result_bed_table = bed_table.concat(new_bed_table)

        self.assertArrayEqual(result_bed_table.subset_by_index(np.array([1, 3, 6])).to_dataframe().values,
                              new_bed_table.to_dataframe().values,
                              )

    def test_merge_sort_bt_return_index(self):
        bed_table = self.__init_test_bed_table()

        new_bed_table = BedTable3()
        new_bed_table.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [2, 80, 50],
            "end": [6, 120, 70],
        }))

        merged_bt, merge_index = BedTable3._merge_sort_bt(bed_table, new_bed_table, return_index=True)

        self.assertEqual(merge_index.shape, (len(bed_table) + len(new_bed_table), 2))
        self.assertArrayEqual(merge_index[:, 0], np.array([0, 1, 0, 1, 0, 0, 1]))
        self.assertArrayEqual(merge_index[:, 1], np.array([0, 0, 1, 1, 2, 3, 2]))

        reconstructed_rows = []
        for source_table, source_idx in merge_index:
            if source_table == 0:
                reconstructed_rows.append(bed_table.get_region_by_index(int(source_idx)).to_dict())
            else:
                reconstructed_rows.append(new_bed_table.get_region_by_index(int(source_idx)).to_dict())

        reconstructed_df = pd.DataFrame(reconstructed_rows, columns=bed_table.column_names)
        self.assertArrayEqual(merged_bt.to_dataframe().values, reconstructed_df.values)
    
    def test_is_sorted(self):
        bed_table = self.__init_test_bed_table()
        bed_table._data_df.loc[0, "chrom"] = "chr21"

        self.assertFalse(bed_table._is_sorted())

        bed_table._sort()

        self.assertTrue(bed_table._is_sorted())

    def test_sort(self):
        bed_table = BedTable3()
        bed_table.load_from_file(self.data_file)

        bed_table._sort()

        self.assertArrayEqual(bed_table.to_dataframe().values,
                              self.data_df.sort_values(["chrom", "start"]).values,
                              )
    
    def test_search_region(self):
        unsorted_bed_path = os.path.join(os.path.dirname(__file__),
                                         "BedTableTestData",
                                         "unsorted_search_region.bed3",
                                         )

        bed_tables = {
            "default": self.__init_test_bed_table(),
            "unsorted_bt": BedTable3(enable_sort=False),
        }
        bed_tables["unsorted_bt"].load_from_file(unsorted_bed_path)
        self.assertFalse(bed_tables["unsorted_bt"]._is_sorted())

        search_cases = pd.DataFrame([
            # default bed table
            {"bed_table": "default", "chrom": "chr2", "start": 3, "end": 7, "overlapping_base": 4},
            {"bed_table": "default", "chrom": "chr2", "start": 5, "end": 12, "overlapping_base": 2},
            {"bed_table": "default", "chrom": "chr2", "start": 9, "end": 12, "overlapping_base": 1},
            {"bed_table": "default", "chrom": "chr1", "start": 4, "end": 10, "overlapping_base": 1},
            {"bed_table": "default", "chrom": "chr3", "start": 1, "end": 10, "overlapping_base": 1},
            {"bed_table": "default", "chrom": "chr1", "start": 6, "end": 7, "overlapping_base": 1},
            {"bed_table": "default", "chrom": "chr2", "start": 7, "end": 8, "overlapping_base": 1},
            {"bed_table": "default", "chrom": "chr2", "start": 3, "end": 7, "overlapping_base": 5},
            # unsorted bed table (enable_sort=False)
            {"bed_table": "unsorted_bt", "chrom": "chr6", "start": 41161904, "end": 41162474, "overlapping_base": 1},
            {"bed_table": "unsorted_bt", "chrom": "chr6", "start": 41073500, "end": 41074000, "overlapping_base": 1},
        ])

        for _, case in search_cases.iterrows():
            bed_table = bed_tables[case["bed_table"]]
            region_idx = bed_table.search_region(case["chrom"],
                                                 int(case["start"]),
                                                 int(case["end"]),
                                                 overlapping_base=int(case["overlapping_base"]),
                                                 )
            expected_idx = self._brute_force_search_region(bed_table,
                                                           case["chrom"],
                                                           int(case["start"]),
                                                           int(case["end"]),
                                                           overlapping_base=int(case["overlapping_base"]),
                                                           )
            self.assertArrayEqual(region_idx, expected_idx)

    def test_search_array(self):
        test_array = np.array(range(10))

        self.assertEqual(BedTable3._search_array(test_array, 5), (4 ,6))
        self.assertEqual(BedTable3._search_array(test_array, 4.5), (4 ,5))

        test_array[5] = 4

        self.assertEqual(BedTable3._search_array(test_array, 4), (3, 6))

        self.assertEqual(BedTable3._search_array(test_array, 100), (9, 10))
        self.assertEqual(BedTable3._search_array(test_array, -1), (-1, 0))

    def __init_test_bed_table(self) -> BedTable3:
        bed_table = BedTable3()
        bed_table.load_from_file(self.data_file)

        return bed_table
    
class TestBedTable3Plus(TestBedTable):
    def setUp(self) -> None:
        super().setUp()
        self.extra_column_names = ["score", "annotation"]
        self.extra_column_dtype = [float, str]
        self.data_df = self.__class__._gen_test_bed_file(self.data_file)

    def tearDown(self) -> None:
        super().tearDown()

    @staticmethod
    def _gen_test_bed_file(bed_path):
        '''
        Generate a test bed file with extra columns.
        Return the data in pd.DataFrame format

        Keyword Arguments:
        - bed_path: str, path to the bed file.

        Return:
        - data_df: pd.DataFrame for data contained in the generated bed.
        '''
        data_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1, 8, 3, 4],
            "end": [5, 12, 7, 8],
            "score": [0.5, 0.8, 0.3, 0.9],
            "annotation": ["gene1", "gene2", "gene3", "gene4"]
        })

        data_df.to_csv(bed_path, 
                       sep="\t", 
                       header=False, 
                       index=False, 
                       )
        
        return data_df

    def test_get_region_extra_column(self):
        bed_table = self.__init_test_bed_table()

        scores = bed_table.get_region_extra_column("score")
        annotations = bed_table.get_region_extra_column("annotation")

        self.assertArrayEqual(scores, self.data_df["score"].values)
        self.assertArrayEqual(annotations, self.data_df["annotation"].values)

    def test_load_from_dataframe(self):
        bed_table = self.__init_test_bed_table()
        bed_table.load_from_dataframe(self.data_df.copy())

        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

    def test_load_from_file(self):
        bed_table = self.__init_test_bed_table()
        bed_table.load_from_file(self.data_file)

        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

    def test_region_subset(self):
        bed_table = self.__init_test_bed_table()

        chrom = "chr2"
        lboundary = 2
        rboundary = 7

        subset_bed_table = bed_table.region_subset(chrom=chrom,
                                                   start=lboundary, 
                                                   end=rboundary,
                                                   )

        expected_df = self.data_df.loc[(self.data_df["chrom"] == chrom) & \
                                      (self.data_df["start"] >= lboundary) & \
                                      (self.data_df["end"] <= rboundary)]

        self.assertArrayEqual(subset_bed_table.to_dataframe().values,
                              expected_df.values)

    def __init_test_bed_table(self):
        bed_table = BedTable3Plus(
            extra_column_names=self.extra_column_names,
            extra_column_dtype=self.extra_column_dtype
        )
        bed_table.load_from_dataframe(self.data_df.copy())
        return bed_table

class TestBedTable6(TestBedTable):
    def setUp(self) -> None:
        super().setUp()
        self.data_df = self.__class__._gen_test_bed_file(self.data_file)

    def tearDown(self) -> None:
        super().tearDown()
    
    @staticmethod
    def _gen_test_bed_file(bed_path):
        '''
        Generate bed file for testing.
        
        Keyword Arguments:
        - bed_path: str, path to the bed file.
        
        Return: 
        - data_df: pd.DataFrame for data contained in the generated bed.
        '''
        data_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1, 8, 3, 4],
            "end": [5, 12, 7, 8],
            "name": ["name1", "name2", "name3", "name4"],
            "score": [0.1, 0.2, 0.3, 0.4],
            "strand": ["+", "-", "+", "-"],
        })

        data_df.to_csv(bed_path, 
                       sep="\t", 
                       header=False, 
                       index=False, 
                       )
        
        return data_df
    
    def test_load_from_file(self):
        bed_table = BedTable6()
        bed_table.load_from_file(self.data_file)

        self.assertArrayEqual(bed_table.to_dataframe().values, 
                              self.data_df.values,
                              )
        
    def test_load_from_dataframe(self):
        bed_table = BedTable6()
        bed_table.load_from_dataframe(self.data_df.copy())

        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

        # Another test case with irregular column names
        irregular_data_df = self.data_df.copy()
        column_map = {col: "col{}".format(i) for i, col in enumerate(irregular_data_df.columns)}
        irregular_data_df.rename(columns=column_map, inplace=True)

        bed_table = BedTable6()
        bed_table.load_from_dataframe(irregular_data_df, 
                                      column_map=column_map,
                                      )
        
        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

        # Test case with None and "." values
        none_data_df = self.data_df.copy().astype("O")
        none_data_df.loc[0, "name"] = None
        none_data_df.loc[1, "score"] = "."
        none_data_df.loc[2, "strand"] = None

        bed_table = BedTable6()
        bed_table.load_from_dataframe(none_data_df)
        bed_table.write(os.path.join(self.data_dir, "bed_table_load_none_test.bed"))

        output_df = pd.read_csv(os.path.join(self.data_dir, "bed_table_load_none_test.bed"),
                                sep="\t",
                                header=None,
                                )

        self.assertEqual(output_df.loc[0, 3], ".")
        self.assertEqual(output_df.loc[1, 4], ".")
        self.assertEqual(output_df.loc[2, 5], ".")

    def test_copy_unstranded_strand(self):
        bed_table = BedTable6()
        bed_table.load_from_dataframe(pd.DataFrame({
            "chrom": ["chr1", "chr2"],
            "start": [1, 3],
            "end": [10, 12],
            "name": ["peak1", "peak2"],
            "score": [100, 200],
            "strand": [".", "."],
        }))

        copied_bed_table = bed_table.copy()
        self.assertEqual(len(copied_bed_table), 2)
        self.assertTrue(copied_bed_table.get_region_strands().isna().all())


    def test_apply_logical_filter(self):
        logical_array = [True, False, True, False]
        logical_array = np.array(logical_array)

        bed_table = self.__init_test_bed_table()

        filtered_bed_table = bed_table.apply_logical_filter(logical_array)

        self.assertArrayEqual(filtered_bed_table.to_dataframe().values, self.data_df.loc[logical_array].values)
    
    def test_get_region_names(self):
        bed_table = self.__init_test_bed_table()

        region_names = bed_table.get_region_names()

        self.assertArrayEqual(region_names, self.data_df["name"].values)

    def test_get_region_scores(self):
        bed_table = self.__init_test_bed_table()

        region_scores = bed_table.get_region_scores()

        self.assertArrayEqual(region_scores, self.data_df["score"].values)

    def test_get_region_strands(self):
        bed_table = self.__init_test_bed_table()

        region_strands = bed_table.get_region_strands()

        self.assertArrayEqual(region_strands, self.data_df["strand"].values)
    
    def test_region_subset(self):
        bed_table = self.__init_test_bed_table()

        chrom = "chr2"
        lboundary = 2
        rboundary = 7

        subset_bed_table = bed_table.region_subset(chrom=chrom,
                                                   start=lboundary, 
                                                   end=rboundary,
                                                   )
        
        subset_df = self.data_df.loc[(self.data_df["chrom"] == chrom) & \
                                     (self.data_df["start"] >= lboundary) & \
                                     (self.data_df["end"] <= rboundary)].copy()

        self.assertArrayEqual(subset_bed_table.to_dataframe().values,
                              subset_df.values, 
                              )

    def __init_test_bed_table(self) -> BedTable6:
        bed_table = BedTable6()
        bed_table.load_from_file(self.data_file)

        return bed_table

class TestBedTable6Plus(TestBedTable):
    def setUp(self) -> None:
        super().setUp()

        self.data_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1, 8, 3, 4],
            "end": [5, 12, 7, 8],
            "name": ["name1", "name2", "name3", "name4"],
            "score": [0.1, 0.2, 0.3, 0.4],
            "strand": ["+", "-", "+", "-"],
            "extra_str_field": ["extra1", "extra2", "extra3", "extra4"],
            "extra_int_field": [1, 2, 3, 4],
        })

        self.extra_field_names = list(self.data_df.columns[6:])
        self.extra_field_dtype = [str, int]

        self.data_df.to_csv(self.data_file, 
                            sep="\t", 
                            header=False, 
                            index=False, 
                            )
        
        self.data_df, self.extra_field_names, self.extra_field_dtype = \
            self._gen_test_bed_file(self.data_file)

    def tearDown(self) -> None:
        return super().tearDown()

    @staticmethod
    def _gen_test_bed_file(bed_path):
        '''
        Generate bed file for testing.
        
        Keyword Arguments:
        - bed_path: str, path to the bed file.
        
        Return:
        - data_df: pd.DataFrame for data contained in the generated bed.
        - extra_field_names: list of extra field names.
        - extra_field_dtype: list of extra field data types.
        '''
        data_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1, 8, 3, 4],
            "end": [5, 12, 7, 8],
            "name": ["name1", "name2", "name3", "name4"],
            "score": [0.1, 0.2, 0.3, 0.4],
            "strand": ["+", "-", "+", "-"],
            "extra_str_field": ["extra1", "extra2", "extra3", "extra4"],
            "extra_int_field": [1, 2, 3, 4],
        })

        extra_field_names = list(data_df.columns[6:])
        extra_field_dtype = [str, int]

        data_df.to_csv(bed_path, 
                       sep="\t", 
                       header=False, 
                       index=False, 
                       )

        return data_df, extra_field_names, extra_field_dtype

    def test_get_region_extra_column(self):
        bed_table = self.__init_test_bed_table()

        for extra_field_name, extra_field_dtype in zip(self.extra_field_names, self.extra_field_dtype):
            extra_field_data = bed_table.get_region_extra_column(extra_field_name)

            self.assertArrayEqual(extra_field_data, self.data_df[extra_field_name].values)

    def test_load_from_dataframe(self):
        bed_table = BedTable6Plus(self.extra_field_names, 
                                  extra_column_dtype=self.extra_field_dtype, 
                                  )
        bed_table.load_from_dataframe(self.data_df.copy())
        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

    def test_load_from_file(self):
        bed_table = BedTable6Plus(self.extra_field_names,
                                  extra_column_dtype=self.extra_field_dtype,
                                  )
        bed_table.load_from_file(self.data_file)

        self.assertArrayEqual(bed_table.to_dataframe().values, self.data_df.values)

    def test_region_subset(self):
        bed_table = self.__init_test_bed_table()

        chrom = "chr2"
        lboundary = 2
        rboundary = 7

        subset_bed_table = bed_table.region_subset(chrom=chrom,
                                                   start=lboundary, 
                                                   end=rboundary,
                                                   )
        
        subset_df = self.data_df.loc[(self.data_df["chrom"] == chrom) & \
                                     (self.data_df["start"] >= lboundary) & \
                                     (self.data_df["end"] <= rboundary)].copy()

        self.assertArrayEqual(subset_bed_table.to_dataframe().values,
                              subset_df.values, 
                              )

    def __init_test_bed_table(self):
        bed_table = BedTable6Plus(extra_column_names=self.extra_field_names,
                                  extra_column_dtype=self.extra_field_dtype, 
                                  )

        bed_table.load_from_file(self.data_file)

        return bed_table

    # TODO: add test for:
    # * __force_dtype


class TestBedTablePairEnd(TestBedTable):
    def setUp(self) -> None:
        super().setUp()

        self.data_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1, 8, 3, 4],
            "end": [5, 12, 7, 8],
            "chrom2": ["chr1", "chr1", "chr2", "chr3"],
            "start2": [6, 13, 8, 5],
            "end2": [7, 14, 9, 8],
            "name": ["name1", "name2", "name3", "name4"],
            "score": [0.1, 0.2, 0.3, 0.4],
            "strand": ["+", "-", "+", "-"],
            "strand2": ["-", "+", "-", "+"],
            "extra_str_field": ["extra1", "extra2", "extra3", "extra4"],
            "extra_int_field": [1, 2, 3, 4],
        })

        self.extra_field_names = ["extra_str_field", "extra_int_field"]
        self.extra_field_dtype = [str, int]
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_get_other_region_chroms(self):
        bed_table = self.__init_test_bed_table()

        chrom2 = bed_table.get_other_region_chroms()

        self.assertArrayEqual(chrom2, self.data_df["chrom2"].values)
    
    def test_get_other_region_starts(self):
        bed_table = self.__init_test_bed_table()

        start2 = bed_table.get_other_region_starts()

        self.assertArrayEqual(start2, self.data_df["start2"].values)

    def test_get_other_region_ends(self):
        bed_table = self.__init_test_bed_table()

        end2 = bed_table.get_other_region_ends()

        self.assertArrayEqual(end2, self.data_df["end2"].values)

    def test_get_pair_names(self):
        bed_table = self.__init_test_bed_table()

        pair_names = bed_table.get_pair_names()

        self.assertArrayEqual(pair_names, self.data_df["name"].values)

    def test_get_pair_scores(self):
        bed_table = self.__init_test_bed_table()

        pair_scores = bed_table.get_pair_scores()

        self.assertArrayEqual(pair_scores, self.data_df["score"].values)

    def test_get_region_strands(self):
        bed_table = self.__init_test_bed_table()

        strands = bed_table.get_region_strands()

        self.assertArrayEqual(strands, self.data_df["strand"].values)
    
    def test_get_other_region_strands(self):
        bed_table = self.__init_test_bed_table()

        strands2 = bed_table.get_other_region_strands()

        self.assertArrayEqual(strands2, self.data_df["strand2"].values)

    def test_get_region_extra_column(self):
        bed_table = self.__init_test_bed_table()

        for extra_field_name, extra_field_dtype in zip(self.extra_field_names, self.extra_field_dtype):
            extra_field_data = bed_table.get_region_extra_column(extra_field_name)

            self.assertArrayEqual(extra_field_data, self.data_df[extra_field_name].values)

    def test_search_pair_extra_column(self):
        bed_table = self.__init_test_bed_table()

        # Perfect match
        extra_str_val = bed_table.search_pair_extra_column(chr1="chr1",
                                                           start1=1,
                                                           end1=5,
                                                           chr2="chr1",
                                                           start2=6,
                                                           end2=7,
                                                           column_name="extra_str_field",
                                                           overlapping_base1=4, 
                                                           overlapping_base2=1, 
                                                           )
        
        self.assertEqual(extra_str_val.max(), "extra1")

        # Non-perfect match
        extra_str_val = bed_table.search_pair_extra_column(chr1="chr1",
                                                           start1=1,
                                                           end1=6,
                                                           chr2="chr1",
                                                           start2=6,
                                                           end2=7,
                                                           column_name="extra_str_field",
                                                           overlapping_base1=5, 
                                                           overlapping_base2=1,
                                                           )

        self.assertArrayEqual(extra_str_val, np.array([]))

        extra_str_val = bed_table.search_pair_extra_column(chr1="chr1",
                                                           start1=1,
                                                           end1=5,
                                                           chr2="chr1",
                                                           start2=6,
                                                           end2=7,
                                                           column_name="extra_int_field",
                                                           overlapping_base1=1, 
                                                           overlapping_base2=1, 
                                                           )

        self.assertEqual(extra_str_val.max(), 1)

        # cross-chromosomal match
        extra_int_val = bed_table.search_pair_extra_column(chr1="chr2",
                                                           start1=6,
                                                           end1=7,
                                                           chr2="chr3",
                                                           start2=7,
                                                           end2=8,
                                                           column_name="extra_int_field",
                                                           overlapping_base1=1, 
                                                           overlapping_base2=1, 
                                                           )
        
        self.assertEqual(extra_int_val.max(), 4)

    def __init_test_bed_table(self):
        bed_table = BedTablePairEnd(extra_column_names=self.extra_field_names,
                                    extra_column_dtype=self.extra_field_dtype,
                                    )
        bed_table.load_from_dataframe(self.data_df.copy())

        return bed_table

if __name__ == "__main__":
    unittest.main()
