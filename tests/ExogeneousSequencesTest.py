
import unittest
import argparse
import shutil
import os

import numpy as np
from Bio import SeqIO

from RGTools.ExogeneousSequences import ExogeneousSequences
from RGTools.GenomicElements import GenomicElements

class TestExogeneousSequences(unittest.TestCase):
    def setUp(self):
        self.__wdir = "ExogeneousSequences_test"

        if not os.path.exists(self.__wdir):
            os.makedirs(self.__wdir)

        self.__fasta_path = os.path.join(self.__wdir, "test.fa")

        with open(self.__fasta_path, "w") as handle:
            handle.write(">chr2:127048023-127048033\nACGTTTTCTG\n")
            handle.write(">chr1:123500-123512\nGTGTAATTACAA\n")
            # record of unequal length
            handle.write(">chr3:123501-123511\nTGTAATTACA\n")

        return super().setUp()
    
    def tearDown(self):
        if os.path.exists(self.__wdir):
            shutil.rmtree(self.__wdir)

    def test_get_region_bed_table(self):
        es = ExogeneousSequences(self.__fasta_path)
        bt = es.get_region_bed_table()
        bt.write(os.path.join(self.__wdir, "test.bed3"))

        ge = GenomicElements(region_file_path=os.path.join(self.__wdir, "test.bed3"),
                             region_file_type="bed3", 
                             fasta_path=self.__fasta_path, 
                             )
        seqs = []
        for region in bt.iter_regions():
            seqs.append(ge.get_region_seq(region["chrom"], region["start"], region["end"]))
        
        self.assertEqual(seqs[0], "ACGTTTTCTG")
    
    def test_get_sequence_ids(self):
        es = ExogeneousSequences(self.__fasta_path)
        self.assertEqual(es.get_sequence_ids()[1], "chr1:123500-123512")
    
    def test_get_all_region_seqs(self):
        es = ExogeneousSequences(self.__fasta_path)
        self.assertEqual(es.get_all_region_seqs()[1], "GTGTAATTACAA")
        self.assertEqual(es.get_all_region_seqs()[2], "TGTAATTACA")

    def test_region_properties_and_lens(self):
        es = ExogeneousSequences(self.__fasta_path)
        self.assertEqual(es.region_file_type, "bed3")
        self.assertEqual(es.get_all_region_lens(), [10, 12, 10])
        with self.assertRaises(NotImplementedError):
            _ = es.region_file_path

    def test_get_all_region_one_hot_requires_homogeneous_lengths(self):
        es = ExogeneousSequences(self.__fasta_path)
        with self.assertRaises(ValueError):
            es.get_all_region_one_hot()

    def test_apply_logical_filter_copies_annotations(self):
        es = ExogeneousSequences(self.__fasta_path)
        stat = np.array([0.1, 0.6, 0.2])
        track = [
            np.arange(10),
            np.arange(12) + 10,
            np.arange(10) + 20,
        ]
        arr_anno = np.arange(18).reshape(3, 2, 3)
        es.load_region_stat_from_arr("stat", stat)
        es.load_region_track_from_list("track", track)
        es.load_region_array_from_arr("arr", arr_anno)

        mask = np.array([True, False, True], dtype=bool)
        new_fa = os.path.join(self.__wdir, "filtered.fa")
        filtered = es.apply_logical_filter(mask, new_fa)

        self.assertTrue(os.path.exists(new_fa))
        self.assertEqual(filtered.get_sequence_ids(), [
            "chr2:127048023-127048033",
            "chr3:123501-123511",
        ])
        self.assertEqual(filtered.get_all_region_lens(), [10, 10])
        np.testing.assert_array_equal(
            filtered.get_stat_arr("stat").reshape(-1,), stat[mask]
        )
        self.assertEqual(filtered.get_anno_type("stat"), "stat")
        self.assertEqual(filtered.get_anno_type("track"), "track")
        filtered_track = filtered.get_track_list("track")
        self.assertEqual(len(filtered_track), 2)
        np.testing.assert_array_equal(filtered_track[0], track[0])
        np.testing.assert_array_equal(filtered_track[1], track[2])
        np.testing.assert_array_equal(filtered.get_arr_anno("arr"), arr_anno[mask])

    def test_write_sequences_to_fasta(self):
        out_fa = os.path.join(self.__wdir, "written.fa")
        seq_ids = ["seqA", "seqB"]
        seqs = ["ACGT", "TTTT"]
        ExogeneousSequences.write_sequences_to_fasta(seq_ids, seqs, out_fa)

        with open(out_fa, "r") as handle:
            parsed = list(SeqIO.parse(handle, "fasta"))
        self.assertEqual([r.id for r in parsed], seq_ids)
        self.assertEqual([str(r.seq) for r in parsed], seqs)

    def test_get_track_list(self):
        es = ExogeneousSequences(self.__fasta_path)
        # Lengths are [10, 12, 10]
        track_list = [np.ones(10), np.zeros(12), np.ones(10) * 2]
        es.load_region_track_from_list("test_track", track_list)
        
        output_list = es.get_track_list("test_track")
        self.assertEqual(len(output_list), 3)
        np.testing.assert_array_equal(output_list[0], track_list[0])
        np.testing.assert_array_equal(output_list[1], track_list[1])
        np.testing.assert_array_equal(output_list[2], track_list[2])
