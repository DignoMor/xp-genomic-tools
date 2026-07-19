
import unittest

from RGTools.utils import reverse_complement

class UtilTest(unittest.TestCase):
    def test_reverse_complement(self):
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("atcg") == "cgat"
        assert reverse_complement("ATCGN") == "NCGAT"
        assert reverse_complement("atcgn") == "ncgat"

