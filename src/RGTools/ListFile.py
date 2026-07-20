
import sys

import numpy as np

class ListFile:
    '''
    Class for reading and handling list files.
    A list file is a simple text file with one item per line.
    '''
    
    def __init__(self, filter_empty_lines=True):
        '''
        Initialize a ListFile instance.

        Keyword arguments:
        - filter_empty_lines: If True, filter out empty lines when reading (default: True).
        '''
        self._contents = []
        self._filter_empty_lines = filter_empty_lines

    @staticmethod
    def write_list_to_file(contents, file_path):
        '''
        Write a python list to a list file (one item per line).

        Keyword arguments:
        - contents: A list-like object containing values to write.
        - file_path: Path to output list file.
                     If file_path is "-" or "stdout", write to stdout.
        '''
        if file_path == "-" or file_path == "stdout":
            file_handle = sys.stdout
        else:
            file_handle = open(file_path, 'w')

        for item in contents:
            file_handle.write(f"{item}\n")

        if file_path != "-" and file_path != "stdout":
            file_handle.close()

    def read_file(self, file_path):
        '''
        Read a list file.

        Keyword arguments:
        - file_path: Path to the list file.
                     If file_path is "-" or "stdin", read from stdin.
        '''
        if file_path == "-" or file_path == "stdin":
            file_handle = sys.stdin
        else:
            file_handle = open(file_path, 'r')

        lines = file_handle.readlines()
        if self._filter_empty_lines:
            self._contents = [line.strip() for line in lines if line.strip()]
        else:
            self._contents = [line.rstrip('\n\r') for line in lines]

        if file_path != "-" and file_path != "stdin":
            file_handle.close()

    def get_contents(self, dtype="str"):
        '''
        Return the contents of the list file as a list of strings.

        Returns:
        - contents: np.array of strings, one per line in the file.
        - dtype: dtype of the outputted array (default: str).
        '''
        return np.array(self._contents, dtype=eval(dtype))

    def get_num_lines(self):
        '''
        Return the number of lines in the list file.

        Returns:
        - num_lines: Number of lines.
        '''
        return len(self._contents)

