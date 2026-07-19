import argparse
import os
import random
import shutil
import unittest

import numpy as np

from ExogeneousSequenceTools.Motif import Motif
from RGTools.ExogeneousSequences import ExogeneousSequences
from RGTools.MemeMotif import MemeMotif

from tests._paths import SAMPLE_MEME

_SKIP_NO_MEME = "Requires tests/fixtures/sample-dna-motif.meme"


@unittest.skipUnless(SAMPLE_MEME.is_file(), _SKIP_NO_MEME)
class MotifTest(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(76)

        self._test_path = "est_motif_test_dir"
        self._meme_motif_path = str(SAMPLE_MEME)
        self._fasta_path = os.path.join(self._test_path, "test_fasta.fasta")

        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)

        meme_motif = MemeMotif(self._meme_motif_path)
        pwm = meme_motif.get_motif_pwm("crp")
        alphabet = meme_motif.get_alphabet()

        seqs = []
        for _ in range(5):
            seq = "ATCG"
            for pos in range(pwm.shape[0]):
                probs = pwm[pos]
                nucleotide = random.choices(alphabet, weights=probs, k=1)[0]
                seq += nucleotide

            seq += "ATCG"
            seqs.append(seq)

        seqs[3] = seqs[3][:-3]

        if os.path.exists(self._fasta_path):
            os.remove(self._fasta_path)

        ExogeneousSequences.write_sequences_to_fasta(
            ["seq1", "seq2", "seq3", "seq4", "seq5"],
            seqs,
            self._fasta_path,
        )

        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(self._test_path):
            shutil.rmtree(self._test_path)

        return super().tearDown()

    def test_motif_search(self):
        args = argparse.Namespace(
            fasta=self._fasta_path,
            motif_file=self._meme_motif_path,
            output_header=os.path.join(self._test_path, "test_motif_search"),
            estimate_background_freq=True,
            reverse_complement=False,
        )

        Motif.motif_search_main(args)
        crp_out = np.load(args.output_header + ".crp.npy")
        self.assertTrue((crp_out[:, 4] > 0).all())
