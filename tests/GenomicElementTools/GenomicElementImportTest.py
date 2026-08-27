import argparse
import os
import shutil
import unittest

import numpy as np

from GenomicElementTools.GenomicElementImport import GenomicElementImport
from RGTools.ExogenousSequences import ExogenousSequences
from RGTools.BedTable import BedTable3

from tests._paths import FIXTURES_DIR


class GenomicElementImportTest(unittest.TestCase):
    def setUp(self):
        self.__bed3_path = str(FIXTURES_DIR / "three_genes.bed3")
        self.__bed6gene_path = str(FIXTURES_DIR / "three_genes.bed6gene")
        self.__wdir = "GenomicElementImport_test"
        if not os.path.exists(self.__wdir):
            os.makedirs(self.__wdir)

        super().setUp()

    def tearDown(self):
        if os.path.exists(self.__wdir):
            shutil.rmtree(self.__wdir)

        super().tearDown()

    def test_import_list_npy(self):
        list_file = os.path.join(self.__wdir, "test_list.txt")
        with open(list_file, "w") as f:
            f.write("region1\n")
            f.write("region2\n")
            f.write("region3\n")

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            inpath=list_file,
            opath=os.path.join(self.__wdir, "test_region_list.npy"),
            informat="stat_list",
            dtype="str",
        )
        GenomicElementImport.import_stat_list(args)

    def test_import_stat_list_main_npy(self):
        list_file = os.path.join(self.__wdir, "test_stat_list.txt")
        with open(list_file, "w") as f:
            f.write("10\n")
            f.write("20\n")
            f.write("30\n")

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            inpath=list_file,
            opath=os.path.join(self.__wdir, "test_stat_list.npy"),
            informat="stat_list",
            dtype="np.int32",
        )
        GenomicElementImport.main(args)

        loaded_arr = np.load(args.opath)
        self.assertEqual(loaded_arr.dtype, np.int32)
        self.assertEqual(len(loaded_arr), 3)
        self.assertEqual(loaded_arr[0], 10)
        self.assertEqual(loaded_arr[1], 20)
        self.assertEqual(loaded_arr[2], 30)

    def test_import_int_list(self):
        list_file = os.path.join(self.__wdir, "test_list.txt")
        with open(list_file, "w") as f:
            f.write("1\n")
            f.write("2\n")
            f.write("3\n")

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            inpath=list_file,
            opath=os.path.join(self.__wdir, "test_region_list.npy"),
            informat="stat_list",
            dtype="np.int32",
        )
        GenomicElementImport.import_stat_list(args)

        loaded_arr = np.load(args.opath)
        self.assertEqual(loaded_arr.dtype, np.int32)
        self.assertEqual(len(loaded_arr), 3)
        self.assertEqual(loaded_arr[0], 1)
        self.assertEqual(loaded_arr[1], 2)
        self.assertEqual(loaded_arr[2], 3)

    def test_import_list_npz(self):
        list_file = os.path.join(self.__wdir, "test_list.txt")
        with open(list_file, "w") as f:
            f.write("region1\n")
            f.write("region2\n")
            f.write("region3\n")

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            inpath=list_file,
            opath=os.path.join(self.__wdir, "test_region_list.npz"),
            informat="stat_list",
            dtype="str",
        )
        GenomicElementImport.import_stat_list(args)

        self.assertTrue(os.path.exists(args.opath))

        loaded_data = np.load(args.opath, allow_pickle=True)
        loaded_arr = loaded_data[loaded_data.files[0]]
        self.assertEqual(len(loaded_arr), 3)
        self.assertEqual(loaded_arr[0], "region1")
        self.assertEqual(loaded_arr[1], "region2")
        self.assertEqual(loaded_arr[2], "region3")

    def test_import_list_mismatch_count(self):
        list_file = os.path.join(self.__wdir, "test_list.txt")
        with open(list_file, "w") as f:
            f.write("region1\n")
            f.write("region2\n")

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            inpath=list_file,
            opath=os.path.join(self.__wdir, "test_region_list.npy"),
            informat="stat_list",
            dtype="str",
        )

        with self.assertRaises(ValueError) as context:
            GenomicElementImport.import_stat_list(args)

        self.assertIn("does not match", str(context.exception))

    def test_import_list_invalid_output_type(self):
        list_file = os.path.join(self.__wdir, "test_list.txt")
        with open(list_file, "w") as f:
            f.write("region1\n")
            f.write("region2\n")
            f.write("region3\n")

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            inpath=list_file,
            opath=os.path.join(self.__wdir, "test_region_list.txt"),
            dtype="str",
            informat="stat_list",
        )

        with self.assertRaises(ValueError) as context:
            GenomicElementImport.import_stat_list(args)

        self.assertIn("Invalid output file type", str(context.exception))

    def test_import_allele_expanded_es_with_stat(self):
        fasta_path = os.path.join(self.__wdir, "allele_expanded.fa")
        ExogenousSequences.write_sequences_to_fasta(
            seq_ids=[
                "chr1_100_110_ref",
                "chr1_100_110_105:A2G",
                "chr1_100_110_106:A2T",
                "chr1_200_210_ref",
                "chr1_200_210_203:C2T",
            ],
            sequences=[
                "AAAAAAAAAA",
                "AAAAAGAAAA",
                "AAAAATAAAA",
                "CCCCCCCCCC",
                "CCCTCCCCCC",
            ],
            fasta_path=fasta_path,
        )

        stat_path = os.path.join(self.__wdir, "signal.npy")
        np.save(stat_path, np.asarray([1.0, 1.2, 0.4, 2.0, 2.5]))

        args = argparse.Namespace(
            inpath=fasta_path,
            anno_oheader=os.path.join(self.__wdir, "allele_import"),
            stat_name=["signal"],
            stat_npy=[stat_path],
            stat_selection_method=["max_abs_fc"],
            informat="allele_expanded_ES",
        )
        GenomicElementImport.import_allele_expanded_es(args)

        out_bed3 = args.anno_oheader + ".bed3"
        out_ref = args.anno_oheader + ".signal.ref.npy"
        out_alt = args.anno_oheader + ".signal.alt.npy"
        self.assertTrue(os.path.exists(out_bed3))
        self.assertTrue(os.path.exists(out_ref))
        self.assertTrue(os.path.exists(out_alt))

        out_bt = BedTable3(enable_sort=False)
        out_bt.load_from_file(out_bed3)
        out_regions = list(out_bt.iter_regions())
        self.assertEqual(len(out_regions), 2)
        self.assertEqual(
            (out_regions[0]["chrom"], out_regions[0]["start"], out_regions[0]["end"]),
            ("chr1", 100, 110),
        )
        self.assertEqual(
            (out_regions[1]["chrom"], out_regions[1]["start"], out_regions[1]["end"]),
            ("chr1", 200, 210),
        )

        ref_arr = np.load(out_ref)
        alt_arr = np.load(out_alt)
        self.assertEqual(ref_arr.shape, (2, 1))
        self.assertEqual(alt_arr.shape, (2, 1))
        self.assertAlmostEqual(ref_arr[0, 0], 1.0)
        self.assertAlmostEqual(ref_arr[1, 0], 2.0)
        self.assertAlmostEqual(alt_arr[0, 0], 0.4)
        self.assertAlmostEqual(alt_arr[1, 0], 2.5)

    def test_import_allele_expanded_es_invalid_fasta_header(self):
        fasta_path = os.path.join(self.__wdir, "invalid.fa")
        ExogenousSequences.write_sequences_to_fasta(
            seq_ids=["chr1:100-110"],
            sequences=["AAAAAAAAAA"],
            fasta_path=fasta_path,
        )

        args = argparse.Namespace(
            inpath=fasta_path,
            anno_oheader=os.path.join(self.__wdir, "invalid_import"),
            stat_name=[],
            stat_npy=[],
            stat_selection_method=[],
            informat="allele_expanded_ES",
        )

        with self.assertRaises(ValueError) as context:
            GenomicElementImport.import_allele_expanded_es(args)
        self.assertIn("Invalid allele_expanded_ES FASTA header", str(context.exception))
