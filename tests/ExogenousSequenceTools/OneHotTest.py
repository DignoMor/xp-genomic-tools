import argparse
import os
import shutil
import unittest

import numpy as np

from ExogenousSequenceTools.cli import ExogenousSequenceTools
from RGTools.ExogenousSequences import ExogenousSequences


class OneHotTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = "est_onehot_test_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        super().setUp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        super().tearDown()

    @staticmethod
    def _write_fasta(path, records):
        seq_ids = [r[0] for r in records]
        seqs = [r[1] for r in records]
        ExogenousSequences.write_sequences_to_fasta(seq_ids, seqs, path)

    def test_onehot_writes_channel_first_output(self):
        fasta_path = os.path.join(self.test_dir, "input.fa")
        output_npy = os.path.join(self.test_dir, "output.npy")
        self._write_fasta(
            fasta_path,
            [
                ("seq1", "ACGT"),
                ("seq2", "TGCA"),
            ],
        )

        args = argparse.Namespace(
            subcommand="onehot",
            fasta=fasta_path,
            opath=output_npy,
        )
        ExogenousSequenceTools.main(args)
        self.assertTrue(os.path.exists(output_npy))

        output_arr = np.load(output_npy)
        self.assertEqual(output_arr.shape, (2, 4, 4))

        np.testing.assert_array_equal(output_arr[0, :, 0], np.array([1, 0, 0, 0], dtype=np.int8))
        np.testing.assert_array_equal(output_arr[0, :, 1], np.array([0, 1, 0, 0], dtype=np.int8))
        np.testing.assert_array_equal(output_arr[0, :, 2], np.array([0, 0, 1, 0], dtype=np.int8))
        np.testing.assert_array_equal(output_arr[0, :, 3], np.array([0, 0, 0, 1], dtype=np.int8))

    def test_onehot_rejects_non_homogeneous_lengths(self):
        fasta_path = os.path.join(self.test_dir, "heterogeneous.fa")
        output_npy = os.path.join(self.test_dir, "output.npy")
        self._write_fasta(
            fasta_path,
            [
                ("seq1", "ACGT"),
                ("seq2", "ACG"),
            ],
        )

        args = argparse.Namespace(
            subcommand="onehot",
            fasta=fasta_path,
            opath=output_npy,
        )
        with self.assertRaises(ValueError):
            ExogenousSequenceTools.main(args)
        self.assertFalse(os.path.exists(output_npy))


if __name__ == "__main__":
    unittest.main()
