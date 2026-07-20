import unittest

import numpy as np

from RGTools.BwTrack import BaseBwTrack, SingleBwTrack, PairedBwTrack

from tests._paths import (
    BW_MN,
    BW_PL,
    HAS_BW_PL,
    HAS_PAIRED_BW,
    SKIP_NO_BW_PL,
    SKIP_NO_PAIRED_BW,
)


class TestBaseBwTrack(unittest.TestCase):
    def test_quantify_signal(self):
        input_signal = np.array([1, 2, 3, 4, 5])

        self.assertEqual(BaseBwTrack.quantify_signal(input_signal, "raw_count"), 
                        input_signal.sum())
        
        self.assertEqual(BaseBwTrack.quantify_signal(input_signal, "RPK"),
                        input_signal.sum() / len(input_signal) * 1e3)

        self.assertTrue((BaseBwTrack.quantify_signal(input_signal, "full_track") == input_signal).all())

    def test_process_padding(self):
        # Test normal padding
        start, end = BaseBwTrack._process_padding("chr1", 100, 200, 10, 20, 50, "raise")
        self.assertEqual(start, 90)
        self.assertEqual(end, 220)

        # Test fallback
        start, end = BaseBwTrack._process_padding("chr1", 100, 200, -100, -100, 50, "fallback")
        self.assertEqual(start, 100)
        self.assertEqual(end, 200)

        # Test drop
        start, end = BaseBwTrack._process_padding("chr1", 100, 200, -100, -100, 50, "drop")
        self.assertIsNone(start)
        self.assertIsNone(end)

        # Test raise
        with self.assertRaises(Exception):
            BaseBwTrack._process_padding("chr1", 100, 200, -100, -100, 50, "raise")


@unittest.skipUnless(HAS_BW_PL, SKIP_NO_BW_PL)
class TestSingleBwTrack(unittest.TestCase):
    def setUp(self):
        self._bw_path = str(BW_PL)

        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_count_single_region(self):
        bw_track = SingleBwTrack(self._bw_path)

        count = bw_track.count_single_region("chr1", 
                                             1000698, 
                                             1000953, 
                                             output_type="raw_count", 
                                             l_pad=0, 
                                             r_pad=0, 
                                             min_len_after_padding=50, 
                                             method_resolving_invalid_padding="raise", 
                                             )

        self.assertEqual(count, 43)
        

@unittest.skipUnless(HAS_PAIRED_BW, SKIP_NO_PAIRED_BW)
class TestPairedBwTrack(unittest.TestCase):
    def setUp(self):
        self._bw_pl_path = str(BW_PL)
        self._bw_mn_path = str(BW_MN)

        super().setUp()

    def tearDown(self):
        pass

    def test_count_single_region(self):
        bw_track = PairedBwTrack(self._bw_pl_path, self._bw_mn_path)

        count = bw_track.count_single_region("chr1", 
                                             1000698, 
                                             1000953, 
                                             strand="+", 
                                             output_type="raw_count", 
                                             l_pad=0, 
                                             r_pad=0, 
                                             min_len_after_padding=50, 
                                             method_resolving_invalid_padding="raise", 
                                             negative_mn=False,
                                             flip_mn=True,
                                             )
        
        self.assertEqual(count, 43)

        count = bw_track.count_single_region("chr1", 
                                             1000698, 
                                             1000953, 
                                             strand="-", 
                                             output_type="RPK", 
                                             l_pad=0, 
                                             r_pad=0, 
                                             min_len_after_padding=50, 
                                             method_resolving_invalid_padding="raise", 
                                             negative_mn=False,
                                             flip_mn=True,
                                             )
        
        self.assertEqual(count, 70 / (1000953 - 1000698) * 1e3)

        count = bw_track.count_single_region("chr1", 
                                             1000698, 
                                             1000953, 
                                             strand="-", 
                                             output_type="full_track", 
                                             l_pad=0, 
                                             r_pad=0, 
                                             min_len_after_padding=1, 
                                             method_resolving_invalid_padding="raise", 
                                             negative_mn=True,
                                             flip_mn=True,
                                             )
        count_no_flip = bw_track.count_single_region("chr1", 
                                                1000698, 
                                                1000953, 
                                                strand="-", 
                                                output_type="full_track", 
                                                l_pad=0, 
                                                r_pad=0, 
                                                min_len_after_padding=1, 
                                                method_resolving_invalid_padding="raise", 
                                                negative_mn=True,
                                                flip_mn=False,
                                                )

        self.assertTrue(count[2] == count_no_flip[-3])


if __name__ == '__main__':
    unittest.main()

