import os
import tempfile
import unittest

from RGTools.ListFile import ListFile


class ListFileTest(unittest.TestCase):
    def test_write_list_to_file_and_read_back(self):
        test_items = ["A", "B", "C"]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name

        ListFile.write_list_to_file(test_items, tmp_path)

        list_file = ListFile()
        list_file.read_file(tmp_path)
        self.assertListEqual(list_file.get_contents(dtype="str").tolist(), test_items)
        self.assertEqual(list_file.get_num_lines(), len(test_items))

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
