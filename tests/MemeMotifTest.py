
import unittest
import shutil
import os

import numpy as np

from RGTools.MemeMotif import MemeMotif
from RGTools.utils import reverse_complement as RC

from tests._paths import SAMPLE_MEME


class TestMemeMotif(unittest.TestCase):
    def setUp(self):
        self._test_dir = "MemeMotif_test"

        if not os.path.exists(self._test_dir):
            os.makedirs(self._test_dir)

        self._meme_file_path = os.path.join(self._test_dir, "test_motifs.meme")
        if not SAMPLE_MEME.is_file():
            self.skipTest(f"Missing MEME fixture: {SAMPLE_MEME}")
        shutil.copyfile(SAMPLE_MEME, self._meme_file_path)

    def tearDown(self):
        if os.path.exists(self._test_dir):
            shutil.rmtree(self._test_dir)

        return super().tearDown()
    
    def test_write_meme_file(self):
        meme = MemeMotif(self._meme_file_path)
        meme.write_meme_file(os.path.join(self._test_dir, "test_motifs_copy.meme"))

        output_meme = MemeMotif(os.path.join(self._test_dir, "test_motifs_copy.meme"))
        self.assertEqual(meme.get_meme_version(), output_meme.get_meme_version())
        self.assertEqual(meme.get_alphabet(), output_meme.get_alphabet())
        self.assertEqual(meme.get_strands(), output_meme.get_strands())
        self.assertEqual(meme.get_bg_freq(), output_meme.get_bg_freq())
        self.assertEqual(meme.get_motif_list(), output_meme.get_motif_list())
        self.assertTrue((meme.get_motif_pwm("crp") == output_meme.get_motif_pwm("crp")).all())
        self.assertEqual(meme.get_motif_alphabet_length("crp"), output_meme.get_motif_alphabet_length("crp"))
        self.assertEqual(meme.get_motif_length("crp"), output_meme.get_motif_length("crp"))
        self.assertEqual(meme.get_motif_num_source_sites("crp"), output_meme.get_motif_num_source_sites("crp"))
        self.assertEqual(meme.get_motif_source_eval("crp"), output_meme.get_motif_source_eval("crp"))

    def test_get_meme_version(self):
        meme = MemeMotif(self._meme_file_path)
        version = meme.get_meme_version()
        self.assertEqual(version, "4")
    
    def test_set_meme_version(self):
        meme = MemeMotif(self._meme_file_path)
        meme.set_meme_version("5")
        self.assertEqual(meme.get_meme_version(), "5")

    def test_get_alphabet(self):
        meme = MemeMotif(self._meme_file_path)
        alphabet = meme.get_alphabet()
        self.assertEqual(alphabet, "ACGT")
    
    def test_set_alphabet(self):
        meme = MemeMotif(self._meme_file_path)
        meme.set_alphabet("ACGTN")
        self.assertEqual(meme.get_alphabet(), "ACGTN")

    def test_get_strands(self):
        meme = MemeMotif(self._meme_file_path)
        strands = meme.get_strands()
        self.assertEqual(len(strands), 2)
        self.assertEqual(strands[0], "+")
        self.assertEqual(strands[1], "-")

    def test_set_strands(self):
        meme = MemeMotif(self._meme_file_path)
        meme.set_strands(["+", "-", "."])
        self.assertEqual(meme.get_strands(), ["+", "-", "."])

    def test_get_bg_freq(self):
        meme = MemeMotif(self._meme_file_path)
        bg_freq = meme.get_bg_freq()
        self.assertEqual(len(bg_freq), 4)
        self.assertAlmostEqual(bg_freq[0], 0.303)
        self.assertAlmostEqual(bg_freq[1], 0.183)
        self.assertAlmostEqual(bg_freq[2], 0.209)
        self.assertAlmostEqual(bg_freq[3], 0.306)
    
    def test_set_bg_freq(self):
        meme = MemeMotif(self._meme_file_path)
        # bg_freq should be a list, not a dict (matching alphabet order: A, C, G, T)
        meme.set_bg_freq([0.25, 0.25, 0.15, 0.35])
        bg_freq = meme.get_bg_freq()
        self.assertEqual(len(bg_freq), 4)
        self.assertAlmostEqual(bg_freq[0], 0.25)  # A
        self.assertAlmostEqual(bg_freq[1], 0.25)  # C
        self.assertAlmostEqual(bg_freq[2], 0.15)  # G
        self.assertAlmostEqual(bg_freq[3], 0.35)  # T

    def test_get_motif_list(self):
        meme = MemeMotif(self._meme_file_path)
        motifs = meme.get_motif_list()
        self.assertEqual(len(motifs), 2)
        self.assertEqual(motifs[0], "crp")
        self.assertEqual(motifs[1], "lexA")
    
    def test_get_motif_pwm(self):
        meme = MemeMotif(self._meme_file_path)
        motif_pwm = meme.get_motif_pwm("crp")
        self.assertEqual(motif_pwm.shape, (19, 4))
        self.assertEqual(motif_pwm[0, 0], 0)
        self.assertEqual(motif_pwm[3, 2], 0.764706)
    
    def test_get_motif_alphabet_length(self):
        meme = MemeMotif(self._meme_file_path)

        alphabet_length = meme.get_motif_alphabet_length("lexA")
        self.assertEqual(alphabet_length, 4)
    
    def test_get_motif_length(self):
        meme = MemeMotif(self._meme_file_path)
        motif_length = meme.get_motif_length("lexA")
        self.assertEqual(motif_length, 18)
    
    def test_get_motif_num_source_sites(self):
        meme = MemeMotif(self._meme_file_path)
        motif_num = meme.get_motif_num_source_sites("lexA")
        self.assertEqual(motif_num, 14)
    
    def test_get_motif_source_eval(self):
        meme = MemeMotif(self._meme_file_path)
        source_eval = meme.get_motif_source_eval("crp")
        self.assertEqual(source_eval, 4.1e-9)
    
    def test_add_motif(self):
        meme = MemeMotif(self._meme_file_path)
        dpe_pwm = np.array([[0.46808, 0.06382, 0.383, 0.0851],
                            [0.0, 0.21, 0.79, 0.0],
                            [0.5745, 0.0425, 0.021, 0.362],
                            [0.043, 0.489, 0.021, 0.447],
                            [0.064, 0.255, 0.66, 0.021],
                            [0.064, 0.1702, 0.1915, 0.5743]])
        
        # Normalize PWM rows to sum to 1.0 (required by documentation and code validation)
        dpe_pwm = dpe_pwm / dpe_pwm.sum(axis=1, keepdims=True)

        meme.add_motif("dpe", {"alphabet_length": 4, 
                               "motif_length": 6, 
                               "num_source_sites": 1000, 
                               "source_eval": 1, 
                               "pwm": dpe_pwm})

        self.assertEqual(meme.get_motif_list(), ["crp", "lexA", "dpe"])
        self.assertEqual(meme.get_motif_alphabet_length("dpe"), 4)
        self.assertEqual(meme.get_motif_length("dpe"), 6)
        self.assertEqual(meme.get_motif_num_source_sites("dpe"), 1000)
        self.assertEqual(meme.get_motif_source_eval("dpe"), 1)
        self.assertTrue(np.allclose(meme.get_motif_pwm("dpe"), dpe_pwm))

    def test_calculate_pwm_score(self):
        meme = MemeMotif(self._meme_file_path)
        motif_pwm = meme.get_motif_pwm("crp")
        score = MemeMotif.calculate_pwm_score("CGCGATCGATCGTTAAGTT", 
                                              motif_pwm, 
                                              bg_freq=meme.get_bg_freq(), 
                                              )
        self.assertAlmostEqual(score, -0.45161817)
        score = MemeMotif.calculate_pwm_score("CGCGATCGATCGTTAAGTT", 
                                              motif_pwm, 
                                              bg_freq=meme.get_bg_freq(), 
                                              reverse_complement=True, 
                                              )
        self.assertTrue(score < -10)

    def test_search_one_motif(self):
        """Test search_one_motif with default strand='+'"""
        meme = MemeMotif(self._meme_file_path)
        score_arr = MemeMotif.search_one_motif("CGCGATCGATCGTTAAGTTG", 
                                               meme.get_alphabet(), 
                                               meme.get_motif_pwm("crp"), 
                                               bg_freq=meme.get_bg_freq(), 
                                               strand="+")
        self.assertEqual(len(score_arr), 20)
        # Last L-1 positions should be padded with minimum score (where L=19)
        motif_len = meme.get_motif_length("crp")
        min_score = score_arr.min()
        self.assertAlmostEqual(score_arr[-motif_len+1:].min(), min_score)
        self.assertAlmostEqual(score_arr[0], -0.45161817)
    
    def test_search_one_motif_reverse_strand(self):
        """Test search_one_motif with strand='-'"""
        meme = MemeMotif(self._meme_file_path)
        test_seq="TGTGATCGAGGTCACACTTATCGAAGTGTGACCTCGATCACA"
        score_arr_rev = MemeMotif.search_one_motif(test_seq, 
                                                   meme.get_alphabet(), 
                                                   meme.get_motif_pwm("crp"), 
                                                   bg_freq=meme.get_bg_freq(), 
                                                   strand="-")
        # Reverse strand should give different scores than forward
        score_arr_fwd = MemeMotif.search_one_motif(test_seq, 
                                                   meme.get_alphabet(), 
                                                   meme.get_motif_pwm("crp"), 
                                                   bg_freq=meme.get_bg_freq(), 
                                                   strand="+")
    
        score_arr_rc_fwd = MemeMotif.search_one_motif(RC(test_seq), 
                                                      meme.get_alphabet(), 
                                                      meme.get_motif_pwm("crp"), 
                                                      bg_freq=meme.get_bg_freq(), 
                                                      strand="+",
                                                      )

        motif_len = meme.get_motif_length("crp")
        self.assertEqual(score_arr_rev[-motif_len], score_arr_rc_fwd[0])
    
    def test_search_one_motif_both_strands(self):
        """Test search_one_motif with strand='both'"""
        meme = MemeMotif(self._meme_file_path)
        score_arr_both = MemeMotif.search_one_motif("CGCGATCGATCGTTAAGTTG", 
                                                    meme.get_alphabet(), 
                                                    meme.get_motif_pwm("crp"), 
                                                    bg_freq=meme.get_bg_freq(), 
                                                    strand="both")
        score_arr_fwd = MemeMotif.search_one_motif("CGCGATCGATCGTTAAGTTG", 
                                                   meme.get_alphabet(), 
                                                   meme.get_motif_pwm("crp"), 
                                                   bg_freq=meme.get_bg_freq(), 
                                                   strand="+")
        score_arr_rev = MemeMotif.search_one_motif("CGCGATCGATCGTTAAGTTG", 
                                                   meme.get_alphabet(), 
                                                   meme.get_motif_pwm("crp"), 
                                                   bg_freq=meme.get_bg_freq(), 
                                                   strand="-")
        # Both should be >= max of forward and reverse at each position
        max_scores = np.maximum(score_arr_fwd, score_arr_rev)
        np.testing.assert_array_almost_equal(score_arr_both, max_scores, decimal=10)
    
    def test_search_one_motif_invalid_strand(self):
        """Test that search_one_motif validates strand parameter"""
        meme = MemeMotif(self._meme_file_path)
        
        with self.assertRaises(ValueError) as context:
            MemeMotif.search_one_motif("CGCGATCGATCGTTAAGTTG", 
                                       meme.get_alphabet(), 
                                       meme.get_motif_pwm("crp"), 
                                       bg_freq=meme.get_bg_freq(), 
                                       strand="invalid")
        
        self.assertIn("strand must be one of", str(context.exception))
        self.assertIn("'+', '-', or 'both'", str(context.exception))
    
    def test_add_motif_pwm_normalization_validation(self):
        """Test that add_motif validates PWM normalization"""
        meme = MemeMotif(self._meme_file_path)
        
        # Create unnormalized PWM (rows don't sum to 1.0)
        unnormalized_pwm = np.array([[0.5, 0.3, 0.2, 0.1],  # sums to 1.1
                                      [0.25, 0.25, 0.25, 0.25]])  # sums to 1.0
        
        with self.assertRaises(ValueError) as context:
            meme.add_motif("test", {
                "alphabet_length": 4,
                "motif_length": 2,
                "num_source_sites": 100,
                "source_eval": 1e-5,
                "pwm": unnormalized_pwm
            })
        
        self.assertIn("PWM rows must sum to 1.0", str(context.exception))
    
    def test_add_motif_pwm_shape_validation(self):
        """Test that add_motif validates PWM shape matches declared dimensions"""
        meme = MemeMotif(self._meme_file_path)
        
        # Create PWM with wrong shape
        wrong_shape_pwm = np.array([[0.25, 0.25, 0.25, 0.25],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.25, 0.25, 0.25, 0.25]])  # 3x4 but declared as 2x4
        
        with self.assertRaises(ValueError) as context:
            meme.add_motif("test", {
                "alphabet_length": 4,
                "motif_length": 2,  # Declared as 2 but PWM has 3 rows
                "num_source_sites": 100,
                "source_eval": 1e-5,
                "pwm": wrong_shape_pwm
            })
        
        self.assertIn("PWM shape", str(context.exception))
        self.assertIn("does not match declared dimensions", str(context.exception))
    
    def test_add_motif_missing_pwm(self):
        """Test that add_motif requires PWM in motif_info"""
        meme = MemeMotif(self._meme_file_path)
        
        with self.assertRaises(ValueError) as context:
            meme.add_motif("test", {
                "alphabet_length": 4,
                "motif_length": 2,
                "num_source_sites": 100,
                "source_eval": 1e-5
                # Missing "pwm" key
            })
        
        self.assertIn("must contain the key 'pwm'", str(context.exception))
    
    def test_calculate_pwm_score_length_mismatch(self):
        """Test that calculate_pwm_score validates sequence length matches PWM"""
        meme = MemeMotif(self._meme_file_path)
        motif_pwm = meme.get_motif_pwm("crp")
        
        # Try with wrong length sequence
        wrong_length_seq = "CGCGATCG"  # Too short
        
        with self.assertRaises(ValueError) as context:
            MemeMotif.calculate_pwm_score(wrong_length_seq, motif_pwm, 
                                         meme.get_alphabet(), meme.get_bg_freq())
        
        self.assertIn("Length of sequence must be the same as the length of the PWM", 
                      str(context.exception))
    
    def test_calculate_pwm_score_default_bg_freq(self):
        """Test that calculate_pwm_score uses uniform bg_freq when None"""
        meme = MemeMotif(self._meme_file_path)
        motif_pwm = meme.get_motif_pwm("crp")
        seq = "CGCGATCGATCGTTAAGTT"  # Correct length
        
        # Score with None bg_freq (should use uniform)
        score_uniform = MemeMotif.calculate_pwm_score(seq, motif_pwm, 
                                                      meme.get_alphabet(), bg_freq=None)
        
        # Score with explicit uniform bg_freq
        uniform_bg = [0.25, 0.25, 0.25, 0.25]
        score_explicit = MemeMotif.calculate_pwm_score(seq, motif_pwm, 
                                                       meme.get_alphabet(), bg_freq=uniform_bg)
        
        self.assertAlmostEqual(score_uniform, score_explicit)
    
    def test_clone_empty(self):
        """Test that clone_empty preserves metadata but not motifs"""
        meme = MemeMotif(self._meme_file_path)
        clone = meme.clone_empty()
        
        # Metadata should be preserved
        self.assertEqual(clone.get_meme_version(), meme.get_meme_version())
        self.assertEqual(clone.get_alphabet(), meme.get_alphabet())
        self.assertEqual(clone.get_strands(), meme.get_strands())
        self.assertEqual(clone.get_bg_freq(), meme.get_bg_freq())
        
        # But no motifs
        self.assertEqual(len(clone.get_motif_list()), 0)
