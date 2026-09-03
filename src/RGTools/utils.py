
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

    if bool_str.upper() == "NONE":
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


def reverse_complement_iupac(seq: str) -> str:
    '''
    Reverse-complement IUPAC DNA while preserving input case.

    Raises ValueError for any non-IUPAC symbol.
    '''
    from Bio.Data.IUPACData import ambiguous_dna_complement

    complement_map = {
        **ambiguous_dna_complement,
        **{k.lower(): v.lower() for k, v in ambiguous_dna_complement.items()},
    }
    try:
        return "".join(complement_map[base] for base in reversed(seq))
    except KeyError as exc:
        raise ValueError(
            f"Cannot reverse-complement non-IUPAC base {exc.args[0]!r}."
        ) from exc


def validate_iupac_dna(seq: str) -> None:
    '''
    Raise ValueError if ``seq`` contains any non-IUPAC DNA symbol.
    '''
    from Bio.Data.IUPACData import ambiguous_dna_complement

    allowed = set(ambiguous_dna_complement) | {
        k.lower() for k in ambiguous_dna_complement
    }
    for base in seq:
        if base not in allowed:
            raise ValueError(f"Non-IUPAC DNA base {base!r}.")


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
