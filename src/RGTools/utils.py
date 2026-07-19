
import numpy as np
import json

def str2bool(bool_str):
    '''
    Convert String to boolean value
    '''

    if not bool_str:
        return False

    if bool_str.upper() == "FALSE":
        return False

    if bool_str.upper() == "None":
        return False

    return True

def str2none(str_val):
    '''
    Convert string to None
    '''

    if str_val.upper() == "NONE":
        return None

    return str_val

def reverse_complement(seq, mapping = {"A": "T", 
                                       "T": "A", 
                                       "C": "G", 
                                       "G": "C", 
                                       "N": "N",
                                       "a": "t",
                                       "t": "a",
                                       "c": "g",
                                       "g": "c",
                                       "n": "n",
                                       }
                       ):
    '''
    Reverse complement a sequence

    Keyword arguments:
    - seq: Sequence to reverse complement
    - mapping: Mapping of bases to their reverse complements
    '''
    return "".join([mapping[base] for base in seq[::-1]])

class NumpyEncoder(json.JSONEncoder):
    '''
    Helper class to encode data with numpy arrays for json serialization.
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)  # Convert NumPy scalars to Python float
        return super().default(obj)
