import argparse
import io
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from ExogeneousSequenceTools.SignalTrack import SignalTrack
from RGTools.ExogeneousSequences import ExogeneousSequences


class SignalTrackTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = "est_signal_track_test_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        super().setUp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        super().tearDown()

    def _create_track_dim_reduction_test_data(self, args):
        tracks = np.zeros((3, 10))
        tracks[0, 3:7] = 1
        tracks[0, 5] = 2
        tracks[0, 7] = 3
        tracks[1, 4] = 1
        tracks[2, 4] = 1
        tracks[2, 5] = 2
        np.save(args.input_npy, tracks)
        return tracks

    def _get_track_dim_reduction_test_args(self, search_range=None):
        return argparse.Namespace(
            input_npy=os.path.join(self.test_dir, "input.npy"),
            output_npy=os.path.join(self.test_dir, "output.npy"),
            search_range=search_range,
        )

    def test_track_dim_reduction_argmax(self):
        args = self._get_track_dim_reduction_test_args()
        args.operation = "argmax"
        self._create_track_dim_reduction_test_data(args)

        SignalTrack.track_dim_reduction_main(args)

        output_track = np.load(args.output_npy)
        self.assertEqual(output_track.shape, (3, 1))
        self.assertEqual(output_track[0, 0], 7)
        self.assertEqual(output_track[1, 0], 4)
        self.assertEqual(output_track[2, 0], 5)

    def test_track_dim_reduction_max(self):
        args = self._get_track_dim_reduction_test_args()
        args.operation = "max"
        self._create_track_dim_reduction_test_data(args)

        SignalTrack.track_dim_reduction_main(args)

        output_track = np.load(args.output_npy)
        self.assertEqual(output_track.shape, (3, 1))
        self.assertEqual(output_track[0, 0], 3)
        self.assertEqual(output_track[1, 0], 1)
        self.assertEqual(output_track[2, 0], 2)

    def test_track_dim_reduction_argmin(self):
        args = self._get_track_dim_reduction_test_args()
        args.operation = "argmin"
        self._create_track_dim_reduction_test_data(args)

        SignalTrack.track_dim_reduction_main(args)

        output_track = np.load(args.output_npy)
        self.assertEqual(output_track.shape, (3, 1))
        self.assertEqual(output_track[0, 0], 0)
        self.assertEqual(output_track[1, 0], 0)
        self.assertEqual(output_track[2, 0], 0)

    def test_track_dim_reduction_min(self):
        args = self._get_track_dim_reduction_test_args()
        args.operation = "min"
        self._create_track_dim_reduction_test_data(args)

        SignalTrack.track_dim_reduction_main(args)

        output_track = np.load(args.output_npy)
        self.assertEqual(output_track.shape, (3, 1))
        self.assertEqual(output_track[0, 0], 0)
        self.assertEqual(output_track[1, 0], 0)
        self.assertEqual(output_track[2, 0], 0)

    def test_track_dim_reduction_with_search_range(self):
        args = self._get_track_dim_reduction_test_args(search_range="2,6")
        args.operation = "argmax"
        self._create_track_dim_reduction_test_data(args)

        SignalTrack.track_dim_reduction_main(args)

        output_track = np.load(args.output_npy)
        self.assertEqual(output_track.shape, (3, 1))
        self.assertEqual(output_track[0, 0], 5)
        self.assertEqual(output_track[1, 0], 4)
        self.assertEqual(output_track[2, 0], 5)

    def test_gen_track_single_loc(self):
        args = argparse.Namespace(
            fasta=os.path.join(self.test_dir, "test.fasta"),
            loc=5,
            output_npy=os.path.join(self.test_dir, "output.npy"),
        )
        args.operation = "single_loc"

        ExogeneousSequences.write_sequences_to_fasta(
            ["seq1", "seq2", "seq3"],
            ["ATCG", "TTGA", "CCAT"],
            args.fasta,
        )

        SignalTrack.gen_track_main(args)

        output_track = np.load(args.output_npy)
        self.assertEqual(output_track.shape, (3, 1))
        self.assertEqual(output_track[0, 0], 5)
        self.assertEqual(output_track[1, 0], 5)
        self.assertEqual(output_track[2, 0], 5)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_print_stat(self, mock_stdout):
        args = argparse.Namespace(
            input_npy=os.path.join(self.test_dir, "input.npy"),
        )
        args.operation = "print_stat"

        np.save(args.input_npy, np.array([1, 2, 3, 4, 5]).reshape(-1, 1))

        SignalTrack.print_stat_main(args)
        self.assertEqual(mock_stdout.getvalue(), "1\n2\n3\n4\n5\n")

        args.input_npy = os.path.join(self.test_dir, "input_signal_track.npy")
        np.save(args.input_npy, np.array([1, 2, 3, 4, 5] * 2).reshape(-1, 2))

        with self.assertRaises(ValueError):
            SignalTrack.print_stat_main(args)
