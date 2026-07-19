import argparse
import contextlib
import io
import os
import shutil
import unittest

import pandas as pd
import numpy as np
import requests

from GenomicElementTools.export import GenomicElementExport
from RGTools.BedTable import BedTable3, BedTable6, BedTable6Plus
from RGTools.ExogeneousSequences import ExogeneousSequences
from RGTools.GenomicElements import GenomicElements

from tests._paths import FIXTURES_DIR, HAS_HG38, HG38_FA, SKIP_NO_HG38


def _ensembl_reachable() -> bool:
    if os.environ.get("RGTOOLS_SKIP_NETWORK_TESTS") == "1":
        return False
    try:
        response = requests.get(
            "https://rest.ensembl.org/info/ping?",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return response.ok
    except requests.RequestException:
        return False


class GenomicElementExportTest(unittest.TestCase):
    def setUp(self):
        self.__bed3_path = str(FIXTURES_DIR / "three_genes.bed3")
        self.__bed6gene_path = str(FIXTURES_DIR / "three_genes.bed6gene")
        self.__fasta_path = str(HG38_FA)
        self.__wdir = "GenomicElementExport_test"
        if not os.path.exists(self.__wdir):
            os.makedirs(self.__wdir)

        self.__sample1_npy_path = os.path.join(self.__wdir, "sample1.npy")
        self.__sample2_npy_path = os.path.join(self.__wdir, "sample2.npy")
        np.save(self.__sample1_npy_path, np.array([1, 2, 3]))
        np.save(self.__sample2_npy_path, np.array([4, 5, 6]))
        self.__mask_npy_path = os.path.join(self.__wdir, "mask.npy")
        np.save(self.__mask_npy_path, np.array([True, False, True]))
        self.__track_npy_path = os.path.join(self.__wdir, "track.npy")
        np.save(self.__track_npy_path, np.array([[1] * 1001, [2] * 1001, [3] * 1001]))
        self.__bed6_path = os.path.join(self.__wdir, "test_snps.bed6")
        bed6_bt = BedTable6(enable_sort=False)
        bed6_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1", "chr1", "chr1"],
                    "start": [161133654, 161136563, 161140556, 161141572],
                    "end": [161133655, 161136564, 161140557, 161141573],
                    "name": ["rs10797093", "rs11265557", "rs12041364", "rs11591206"],
                    "score": [2.677e-08, 2.242e-08, 2.206e-08, 2.326e-08],
                    "strand": ["+", "+", "+", "+"],
                }
            )
        )
        bed6_bt.write(self.__bed6_path)

        self.__pl_track_path = os.path.join(self.__wdir, "pl_track.npz")
        self.__mn_track_path = os.path.join(self.__wdir, "mn_track.npz")

        pl_track = np.zeros((3, 1001))
        pl_track[0, 100] = 10.0
        pl_track[1, 200] = 10.0
        pl_track[2, 50] = 10.0

        mn_track = np.zeros((3, 1001))
        mn_track[0, 300] = 10.0
        mn_track[1, 150] = 10.0
        mn_track[2, 250] = 10.0

        np.savez(self.__pl_track_path, pl_track)
        np.savez(self.__mn_track_path, mn_track)

        self.__chrom_size_path = os.path.join(self.__wdir, "test.chrom.sizes")
        self.__chrom_size_df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr3", "chr4", "chrFake", "chr6"],
                "size": [248956422, 242193529, 198295559, 190214555, 1000000, 170805979],
            },
            columns=["chrom", "size"],
        )
        self.__chrom_size_df.to_csv(
            self.__chrom_size_path,
            sep="\t",
            header=False,
            index=False,
        )

        super().setUp()

    def tearDown(self):
        if os.path.exists(self.__wdir):
            shutil.rmtree(self.__wdir)

        super().tearDown()

    def _require_hg38(self):
        if not HAS_HG38:
            self.skipTest(SKIP_NO_HG38)

    def __read_fasta_records(self, fasta_path):
        es = ExogeneousSequences(fasta_path)
        return list(zip(es.get_sequence_ids(), es.get_all_region_seqs()))

    def test_export_exogeneous_sequences(self):
        self._require_hg38()
        ofile = os.path.join(self.__wdir, "test.fa")
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            fasta_path=self.__fasta_path,
            opath=ofile,
            oformat="ExogeneousSequences",
        )
        GenomicElementExport.export_exogeneous_sequences(args)

        with open(ofile, "r") as handle:
            lines = handle.readlines()
            self.assertEqual(len(lines), 6)
            self.assertEqual(lines[1][:10], "CCCCATCCCC")

    def test_export_stat_list(self):
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            stat_npy=self.__sample1_npy_path,
            opath=os.path.join(self.__wdir, "test.stat_list.txt"),
            dtype="np.int64",
            oformat="stat_list",
        )
        GenomicElementExport.export_stat_list(args)

        with open(args.opath, "r") as handle:
            lines = [line.strip() for line in handle.readlines()]

        self.assertEqual(lines, ["1", "2", "3"])

    def test_export_stat_list_stdout(self):
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            stat_npy=self.__sample1_npy_path,
            opath="-",
            dtype="np.int64",
            oformat="stat_list",
        )
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            GenomicElementExport.export_stat_list(args)

        self.assertEqual(stdout_buffer.getvalue(), "1\n2\n3\n")

    def test_export_wtes(self):
        self._require_hg38()
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            fasta_path=self.__fasta_path,
            num_replicates=2,
            opath=os.path.join(self.__wdir, "test.wtes.fa"),
            oformat="WTES",
        )
        GenomicElementExport.export_wtes(args)

        with open(args.opath, "r") as handle:
            lines = [line.strip() for line in handle.readlines()]

        self.assertEqual(len(lines), 12)
        self.assertEqual(lines[0], ">chr14:75278325-75279326_0")
        self.assertEqual(lines[2], ">chr14:75278325-75279326_1")
        self.assertEqual(lines[4], ">chr17:45894026-45895027_0")
        self.assertEqual(lines[10], ">chr6:170553801-170554802_1")
        self.assertEqual(lines[1][:10], "CCCCATCCCC")
        self.assertEqual(lines[1], lines[3])
        self.assertEqual(lines[5], lines[7])
        self.assertEqual(lines[9], lines[11])

    def test_export_count_table(self):
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            opath=os.path.join(self.__wdir, "test.count.csv"),
            oformat="CountTable",
            region_id_type="default",
            sample_name=["sample1", "sample2"],
            stat_npy=[self.__sample1_npy_path, self.__sample2_npy_path],
        )
        GenomicElementExport.export_count_table(args)

        output_df = pd.read_csv(
            args.opath,
            index_col=0,
        )

        self.assertEqual(output_df.shape, (3, 2))
        self.assertEqual(output_df.iloc[0, 0], 1)
        self.assertEqual(output_df.iloc[1, 0], 2)
        self.assertEqual(output_df.iloc[2, 0], 3)

        args.region_file_path = self.__bed6gene_path
        args.region_file_type = "bed6gene"
        args.region_id_type = "gene_symbol"
        GenomicElementExport.export_count_table(args)

        output_df = pd.read_csv(
            args.opath,
            index_col=0,
        )

        self.assertEqual(output_df.shape, (3, 2))
        self.assertEqual(output_df.iloc[0, 0], 1)
        self.assertEqual(output_df.iloc[2, 0], 3)
        self.assertEqual(output_df.index[1], "gene3")

    def test_export_chrom_filtered_ge(self):
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            chrom_size=self.__chrom_size_path,
            opath=os.path.join(self.__wdir, "test.chrom.filtered.ge"),
            oformat="ChromFilteredGE",
        )
        GenomicElementExport.export_chrom_filtered_ge(args)

        output_bt = BedTable3()
        output_bt.load_from_file(args.opath)
        self.assertEqual(len(output_bt), 1)

    def test_export_masked_ge(self):
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            mask_npy=self.__mask_npy_path,
            opath=os.path.join(self.__wdir, "test.masked.bed3"),
            anno_name=["sample1", "track1"],
            anno_npy=[self.__sample1_npy_path, self.__track_npy_path],
            anno_type=["stat", "track"],
            anno_oheader=os.path.join(self.__wdir, "masked_output"),
            oformat="MaskedGE",
        )
        GenomicElementExport.export_masked_ge(args)

        output_bt = BedTable3(enable_sort=False)
        output_bt.load_from_file(args.opath)
        output_regions = list(output_bt.iter_regions())

        self.assertEqual(len(output_bt), 2)
        self.assertEqual(output_regions[0]["chrom"], "chr14")
        self.assertEqual(output_regions[1]["chrom"], "chr6")

        masked_stat = np.load(args.anno_oheader + ".sample1.npy")
        masked_track = np.load(args.anno_oheader + ".track1.npy")
        np.testing.assert_array_equal(masked_stat.reshape(-1,), np.array([1, 3]))
        np.testing.assert_array_equal(masked_track[:, 0], np.array([1, 3]))

    def test_export_trebed(self):
        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            pl_sig_track=self.__pl_track_path,
            mn_sig_track=self.__mn_track_path,
            opath=os.path.join(self.__wdir, "test.trebed"),
            oformat="TREbed",
        )
        GenomicElementExport.export_trebed(args)

        trebed_bt = GenomicElements.BedTableTREBed(enable_sort=False)
        trebed_bt.load_from_file(args.opath)

        self.assertEqual(len(trebed_bt), 3)

        regions = list(trebed_bt.iter_regions())

        self.assertEqual(regions[0]["chrom"], "chr14")
        self.assertEqual(regions[0]["start"], 75278325)
        self.assertEqual(regions[0]["end"], 75279326)
        self.assertEqual(regions[0]["name"], "chr14:75278325-75279326")
        self.assertEqual(regions[0]["fwdTSS"], 75278325 + 100)
        self.assertEqual(regions[0]["revTSS"], 75278325 + 300)

        self.assertEqual(regions[1]["chrom"], "chr17")
        self.assertEqual(regions[1]["start"], 45894026)
        self.assertEqual(regions[1]["end"], 45895027)
        self.assertEqual(regions[1]["fwdTSS"], 45894026 + 200)
        self.assertEqual(regions[1]["revTSS"], 45894026 + 150)

        self.assertEqual(regions[2]["chrom"], "chr6")
        self.assertEqual(regions[2]["start"], 170553801)
        self.assertEqual(regions[2]["end"], 170554802)
        self.assertEqual(regions[2]["fwdTSS"], 170553801 + 50)
        self.assertEqual(regions[2]["revTSS"], 170553801 + 250)

    def test_export_trebed_negative_track(self):
        neg_mn_track_path = os.path.join(self.__wdir, "neg_mn_track.npz")

        mn_track = np.zeros((3, 1001))
        mn_track[0, 300] = -15.0
        mn_track[1, 150] = -20.0
        mn_track[2, 250] = -10.0

        np.savez(neg_mn_track_path, mn_track)

        args = argparse.Namespace(
            region_file_path=self.__bed3_path,
            region_file_type="bed3",
            pl_sig_track=self.__pl_track_path,
            mn_sig_track=neg_mn_track_path,
            opath=os.path.join(self.__wdir, "test_neg.trebed"),
            oformat="TREbed",
        )
        GenomicElementExport.export_trebed(args)

        trebed_bt = GenomicElements.BedTableTREBed(enable_sort=False)
        trebed_bt.load_from_file(args.opath)
        regions = list(trebed_bt.iter_regions())

        self.assertEqual(regions[0]["revTSS"], 75278325 + 300)

    def test_export_merged_ge(self):
        left_region_path = os.path.join(self.__wdir, "merge_left.bed3")
        right_region_path = os.path.join(self.__wdir, "merge_right.bed3")

        left_bt = BedTable3(enable_sort=False)
        left_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr2", "chr2"],
                    "start": [100, 200],
                    "end": [110, 210],
                }
            )
        )
        left_bt.write(left_region_path)

        right_bt = BedTable3(enable_sort=False)
        right_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1"],
                    "start": [50, 300],
                    "end": [60, 310],
                }
            )
        )
        right_bt.write(right_region_path)

        left_stat_path = os.path.join(self.__wdir, "left.stat.npy")
        right_stat_path = os.path.join(self.__wdir, "right.stat.npy")
        left_track_path = os.path.join(self.__wdir, "left.track.npy")
        right_track_path = os.path.join(self.__wdir, "right.track.npy")
        np.save(left_stat_path, np.array([100, 200]))
        np.save(right_stat_path, np.array([10, 20]))
        np.save(left_track_path, np.array([[1] * 10, [2] * 10]))
        np.save(right_track_path, np.array([[3] * 10, [4] * 10]))

        oheader = os.path.join(self.__wdir, "merged")
        args = argparse.Namespace(
            left_region_file_path=left_region_path,
            right_region_file_path=right_region_path,
            region_file_type="bed3",
            anno_name=["stat", "track"],
            left_anno_path=[left_stat_path, left_track_path],
            right_anno_path=[right_stat_path, right_track_path],
            anno_type=["stat", "track"],
            oheader=oheader,
            oformat="MergedGE",
        )
        GenomicElementExport.export_merged_ge(args)

        merged_region_path = oheader + ".bed3"
        merged_bt = BedTable3()
        merged_bt.load_from_file(merged_region_path)
        merged_df = merged_bt.to_dataframe()
        self.assertEqual(merged_df.iloc[0]["chrom"], "chr1")
        self.assertEqual(merged_df.iloc[1]["chrom"], "chr1")
        self.assertEqual(merged_df.iloc[2]["chrom"], "chr2")
        self.assertEqual(merged_df.iloc[3]["chrom"], "chr2")

        stat_arr = np.load(oheader + ".stat.npy")
        track_arr = np.load(oheader + ".track.npy")
        np.testing.assert_array_equal(stat_arr.reshape(-1,), np.array([10, 20, 100, 200]))
        np.testing.assert_array_equal(track_arr[:, 0], np.array([3, 4, 1, 2]))

    @unittest.skipUnless(
        _ensembl_reachable(),
        "Ensembl REST API unreachable or RGTOOLS_SKIP_NETWORK_TESTS=1",
    )
    def test_export_bed6poly(self):
        args = argparse.Namespace(
            region_file_path=self.__bed6_path,
            region_file_type="bed6",
            genome_version="hg38",
            opath=os.path.join(self.__wdir, "test.bed6poly"),
            oformat="bed6poly",
        )

        GenomicElementExport.export_bed6poly(args)

        output_bt = BedTable6Plus(
            extra_column_names=["polymorphism"],
            extra_column_dtype=[str],
            enable_sort=False,
        )
        output_bt.load_from_file(args.opath)
        regions = list(output_bt.iter_regions())

        self.assertEqual(len(regions), 4)
        expected = {
            "rs10797093": "T/G",
            "rs11265557": "T/C/G",
            "rs12041364": "G/A/C",
            "rs11591206": "C/A/G/T",
        }
        for region in regions:
            self.assertIn(region["name"], expected)
            self.assertEqual(str(region["polymorphism"]), expected[region["name"]])

    def test_export_allele_expanded_es(self):
        self._require_hg38()
        tre_path = os.path.join(self.__wdir, "single_tre.bed3")
        snp_path = os.path.join(self.__wdir, "single_tre.snp.bed6plus")
        ofa = os.path.join(self.__wdir, "single_tre.allele_expanded.fa")

        tre_bt = BedTable3(enable_sort=False)
        tre_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr14"],
                    "start": [75278325],
                    "end": [75279326],
                }
            )
        )
        tre_bt.write(tre_path)

        ge = GenomicElements(tre_path, "bed3", self.__fasta_path)
        ref_seq = ge.get_all_region_seqs()[0]
        mut_index = 10
        ref_base = ref_seq[mut_index].upper()
        alt_base = "A" if ref_base != "A" else "C"

        snp_bt = BedTable6Plus(
            extra_column_names=["bases"],
            extra_column_dtype=[str],
            enable_sort=False,
        )
        snp_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr14"],
                    "start": [75278325 + mut_index],
                    "end": [75278325 + mut_index + 1],
                    "name": ["rsDraft1"],
                    "score": [0.0],
                    "strand": ["+"],
                    "bases": [f"{ref_base}/{alt_base}"],
                }
            )
        )
        snp_bt.write(snp_path)

        args = argparse.Namespace(
            region_file_path=tre_path,
            region_file_type="bed3",
            fasta_path=self.__fasta_path,
            inpath_polymorphisms=snp_path,
            opath=ofa,
            oformat="allele_expanded_ES",
        )

        GenomicElementExport.export_allele_expanded_es(args)
        records = self.__read_fasta_records(ofa)

        self.assertEqual(len(records), 2)
        ref_id, ref_out_seq = records[0]
        mut_id, mut_out_seq = records[1]
        self.assertTrue(ref_id.endswith("_ref"))
        self.assertIn(f"{75278325 + mut_index}:{ref_base}2{alt_base}", mut_id)
        self.assertEqual(ref_out_seq[mut_index].upper(), ref_base)
        self.assertEqual(mut_out_seq[mut_index].upper(), alt_base)

    def test_export_allele_expanded_es_skips_ref_and_multibase_alt(self):
        self._require_hg38()
        tre_path = os.path.join(self.__wdir, "single_tre2.bed3")
        snp_path = os.path.join(self.__wdir, "single_tre2.snp.bed6plus")
        ofa = os.path.join(self.__wdir, "single_tre2.allele_expanded.fa")

        tre_bt = BedTable3(enable_sort=False)
        tre_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr14"],
                    "start": [75278325],
                    "end": [75279326],
                }
            )
        )
        tre_bt.write(tre_path)

        ge = GenomicElements(tre_path, "bed3", self.__fasta_path)
        ref_seq = ge.get_all_region_seqs()[0]
        mut_index = 20
        ref_base = ref_seq[mut_index].upper()
        alt_base = "G" if ref_base != "G" else "T"

        snp_bt = BedTable6Plus(
            extra_column_names=["bases"],
            extra_column_dtype=[str],
            enable_sort=False,
        )
        snp_bt.load_from_dataframe(
            pd.DataFrame(
                {
                    "chrom": ["chr14"],
                    "start": [75278325 + mut_index],
                    "end": [75278325 + mut_index + 1],
                    "name": ["rsDraft2"],
                    "score": [0.0],
                    "strand": ["+"],
                    "bases": [f"{ref_base}/{alt_base}/TT"],
                }
            )
        )
        snp_bt.write(snp_path)

        args = argparse.Namespace(
            region_file_path=tre_path,
            region_file_type="bed3",
            fasta_path=self.__fasta_path,
            inpath_polymorphisms=snp_path,
            opath=ofa,
            oformat="allele_expanded_ES",
        )

        GenomicElementExport.export_allele_expanded_es(args)
        records = self.__read_fasta_records(ofa)

        self.assertEqual(len(records), 2)
        self.assertTrue(records[0][0].endswith("_ref"))
        self.assertIn(f"{75278325 + mut_index}:{ref_base}2{alt_base}", records[1][0])
        self.assertEqual(records[1][1][mut_index].upper(), alt_base)
