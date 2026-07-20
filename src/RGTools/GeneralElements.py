import abc
import numpy as np

from Bio import SeqIO

class GeneralElements(abc.ABC):
    '''
    Abstract base class for GenomicElements and ExogeneousSequences.
    
    This class provides common functionality for handling genomic elements
    and exogeneous sequences, including:
    - Annotation management
    - Sequence operations
    - Bed table operations
    - Common utility methods
    '''

    def __init__(self):
        '''
        Initialize common attributes for all element types.
        '''
        self._anno_arr_dict = {}
        self._anno_length_dict = {}
        self._anno_type_dict = {}
        self._genome_index = None

    @property
    @abc.abstractmethod
    def fasta_path(self):
        '''
        Abstract property for the path to the fasta file.
        Must be implemented by subclasses.
        '''
        pass

    @property
    @abc.abstractmethod
    def region_file_type(self):
        '''
        Abstract property for the region file type.
        Must be implemented by subclasses.
        '''
        pass

    @property
    @abc.abstractmethod
    def region_file_path(self):
        '''
        Abstract property for the path to the region file.
        Must be implemented by subclasses.
        '''
        pass

    @abc.abstractmethod
    def get_region_bed_table(self):
        '''
        Abstract method to return a bed table object for the regions.
        Must be implemented by subclasses.
        '''
        pass

    @abc.abstractmethod
    def get_all_region_seqs(self):
        '''
        Get the sequences for all regions.
        Must be implemented by subclasses to choose an appropriate strategy.

        Returns:
        - A list of strings of length region_length
        '''
        pass

    def _get_genome_index(self):
        if self._genome_index is None:
            self._genome_index = SeqIO.index(self.fasta_path, "fasta")
        return self._genome_index

    def close(self):
        '''
        Close cached resources held by this instance.
        '''
        if self._genome_index is not None:
            self._genome_index.close()
            self._genome_index = None

    def __del__(self):
        # Best-effort cleanup for cached FASTA index handles.
        try:
            self.close()
        except Exception:
            pass

    def get_region_seq(self, chrom: str, start: int, end: int, index_genome: bool = True):
        '''
        Return the sequence of a given region.
        The coordinates are of bed convention.

        Keyword arguments:
        - chrom: Chromosome name.
        - start: Start coordinate.
        - end: End coordinate.
        - index_genome: Whether to use indexed access to the fasta file.

        Returns:
        - seq: Sequence of the region, or None if chromosome not found.
        '''
        if index_genome:
            genome_index = self._get_genome_index()
            if chrom in genome_index:
                return str(genome_index[chrom].seq[start:end])  # Convert to 0-based index
            return None

        with open(self.fasta_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if record.id == chrom:
                    return str(record.seq[start:end])  # Convert to 0-based index
        return None

    def get_region_lens(self):
        '''
        Return an array of per-region lengths.
        '''
        bed_table = self.get_region_bed_table()
        return bed_table.get_end_locs() - bed_table.get_start_locs()

    def get_all_region_one_hot(self):
        '''
        Get the one hot encoding for all regions.
        This function reads the genome file to memory 
        so that it is memory intensive.
        '''
        out_seqs = self.get_all_region_seqs()

        out_seq_lens = np.array([len(s) for s in out_seqs])
        if not (out_seq_lens == out_seq_lens[0]).all():
            raise ValueError(f"Regions are not length-homogeneous; lengths={out_seq_lens.tolist()}")
        
        out_arr = np.zeros((len(out_seqs), out_seq_lens[0], 4), dtype="int8")
        for i, seq in enumerate(out_seqs):
            out_arr[i] = self.one_hot_encoding(seq)
        return out_arr

    @abc.abstractmethod
    def apply_logical_filter(self, logical, new_path):
        '''
        Abstract method to apply logical filter to the regions.
        Must be implemented by subclasses.
        '''
        pass

    def get_num_regions(self):
        '''
        Return the number of regions.
        '''
        return self.get_region_bed_table().__len__()

    def load_region_anno_from_npy(self, anno_name, npy_path, anno_type="array"):
        '''
        Load annotation for each element in the region file. 
        
        Keyword arguments:
        - anno_name: Name of the annotation.
        - npy_path: Path to the numpy file (.npy or .npz format).

        Returns:
        None
        '''
        data = np.load(npy_path, allow_pickle=False)
        # Check if it's an npz file (has 'files' attribute)
        if hasattr(data, 'files'):
            # .npz file - extract the array
            keys = list(data.keys())
            if len(keys) == 1:
                anno_arr = data[keys[0]]
            else:
                raise ValueError(f"NPZ file {npy_path} contains multiple arrays ({len(keys)}). "
                               f"Please use .npy format or specify which array to use. "
                               f"Available keys: {keys}")
        else:
            # .npy file - data is already the array
            anno_arr = data
        
        if anno_type == "track":
            anno_arr = np.asarray(anno_arr)
            if len(anno_arr.shape) != 2:
                raise ValueError(f"Track annotation must be 2D; got shape {anno_arr.shape}")
            if anno_arr.shape[0] != self.get_num_regions():
                raise ValueError(
                    f"Track annotation shape {anno_arr.shape} does not match number of regions {self.get_num_regions()}"
                )
            max_len = int(self.get_region_lens().max())
            if anno_arr.shape[1] != max_len:
                raise ValueError(f"Track annotation width {anno_arr.shape[1]} must equal max region length {max_len}")
            self._anno_arr_dict[anno_name] = anno_arr
            self._anno_length_dict[anno_name] = max_len
            self._anno_type_dict[anno_name] = "track"
        elif anno_type == "stat":
            self.load_region_stat_from_arr(anno_name, anno_arr)
        elif anno_type == "mask":
            self.load_mask_from_arr(anno_name, anno_arr)
        elif anno_type == "array":
            self.load_region_array_from_arr(anno_name, anno_arr)
        else:
            raise ValueError(
                f"Unsupported anno_type '{anno_type}'. "
                "Use one of: 'track', 'stat', 'mask', 'array'."
            )

    def load_region_track_from_list(self, anno_name, anno_list):
        '''
        Load per-region annotation track from a python list.
        Useful when loading annotations for non-length-homogeneous elements.
        
        Keyword arguments:
        - anno_name: Name of the annotation.
        - anno_list: List of numpy arrays, one for each region.
        '''
        if len(anno_list) != self.get_num_regions():
            raise ValueError(f"List length {len(anno_list)} does not match number of regions {self.get_num_regions()}")
        
        region_lens = self.get_region_lens()
        for i, (anno, region_len) in enumerate(zip(anno_list, region_lens)):
            if len(anno) != region_len:
                raise ValueError(f"Annotation length at index {i} ({len(anno)}) does not match region length ({region_len})")
        
        max_len = int(region_lens.max())
        anno_arr = np.zeros((len(anno_list), max_len))
        for i, anno in enumerate(anno_list):
            anno_arr[i, :len(anno)] = anno
            
        self._anno_arr_dict[anno_name] = anno_arr
        self._anno_length_dict[anno_name] = max_len
        self._anno_type_dict[anno_name] = "track"

    def load_region_stat_from_arr(self, anno_name, anno_arr):
        '''
        Load per-region annotation stat from a numpy array.
        
        Keyword arguments:
        - anno_name: Name of the annotation.
        - anno_arr: numpy array of shape (N,) or (N, 1) containing stat values.
        '''
        anno_arr = np.asarray(anno_arr)
        if anno_arr.shape[0] != self.get_num_regions():
            raise ValueError(f"Array length {anno_arr.shape[0]} does not match number of regions {self.get_num_regions()}")
        
        # Reshape to (N, 1) if needed
        if len(anno_arr.shape) == 1:
            anno_arr = anno_arr.reshape(-1, 1)
        elif len(anno_arr.shape) == 2 and anno_arr.shape[1] == 1:
            pass  # Already correct shape
        else:
            raise ValueError(f"Stat array must be 1D or 2D with shape (N, 1); got shape {anno_arr.shape}")
        
        self._anno_arr_dict[anno_name] = anno_arr
        self._anno_length_dict[anno_name] = 1
        self._anno_type_dict[anno_name] = "stat"

    def load_mask_from_arr(self, anno_name, anno_arr):
        '''
        Load per-region mask from a numpy array.
        
        Keyword arguments:
        - anno_name: Name of the annotation.
        - anno_arr: numpy array of shape (N,) or (N, 1) containing boolean values.
        '''
        anno_arr = np.asarray(anno_arr)
        if anno_arr.dtype != np.bool_:
            raise ValueError(
                f"Mask array must have boolean dtype; got dtype {anno_arr.dtype}"
            )
        if anno_arr.shape[0] != self.get_num_regions():
            raise ValueError(f"Array length {anno_arr.shape[0]} does not match number of regions {self.get_num_regions()}")
        
        # Reshape to (N, 1) if needed
        if len(anno_arr.shape) == 1:
            anno_arr = anno_arr.reshape(-1, 1)
        elif len(anno_arr.shape) == 2 and anno_arr.shape[1] == 1:
            pass  # Already correct shape
        else:
            raise ValueError(f"Mask array must be 1D or 2D with shape (N, 1); got shape {anno_arr.shape}")
        
        self._anno_arr_dict[anno_name] = anno_arr
        self._anno_length_dict[anno_name] = 1
        self._anno_type_dict[anno_name] = "mask"

    def load_region_array_from_arr(self, anno_name, anno_arr):
        '''
        Load per-region array annotation from a numpy array.

        Keyword arguments:
        - anno_name: Name of the annotation.
        - anno_arr: numpy array with shape (N, ...).
        '''
        anno_arr = np.asarray(anno_arr)
        if len(anno_arr.shape) < 2:
            raise ValueError(
                f"Array annotation must have at least 2 dimensions (N, ...); got shape {anno_arr.shape}"
            )
        if anno_arr.shape[0] != self.get_num_regions():
            raise ValueError(
                f"Array annotation shape {anno_arr.shape} does not match number of regions {self.get_num_regions()}"
            )

        self._anno_arr_dict[anno_name] = anno_arr
        self._anno_length_dict[anno_name] = tuple(anno_arr.shape[1:])
        self._anno_type_dict[anno_name] = "array"

    def get_anno_dim(self, anno_name):
        '''
        Return the dimension of the annotation.

        Keyword arguments:
        - anno_name: Name of the annotation.

        Return: 
        - dim: Dimension of the annotation.
        '''
        if anno_name not in self._anno_length_dict:
            raise ValueError(f"Annotation '{anno_name}' not found. Available annotations: {list(self._anno_length_dict.keys())}")
        return self._anno_length_dict[anno_name]

    def get_anno_type(self, anno_name):
        '''
        Return the annotation type: "stat", "track", "mask", or "array".
        '''
        return self._anno_type_dict[anno_name]

    def get_track_list(self, anno_name):
        '''
        Return a list of track annotations sliced to each region length.
        '''
        if self.get_anno_type(anno_name) != "track":
            raise ValueError(
                f"Annotation '{anno_name}' is not a track. Type={self.get_anno_type(anno_name)}"
            )
        anno_arr = self._anno_arr_dict[anno_name]
        region_lens = self.get_region_lens()
        return [anno_arr[i, :region_lens[i]] for i in range(self.get_num_regions())]

    def get_stat_arr(self, anno_name):
        '''
        Return stat annotation array with shape (N, 1).
        '''
        if self.get_anno_type(anno_name) != "stat":
            raise ValueError(
                f"Annotation '{anno_name}' is not a stat. Type={self.get_anno_type(anno_name)}"
            )
        return self._anno_arr_dict[anno_name]

    def get_mask_arr(self, anno_name):
        '''
        Return mask annotation array with shape (N, 1).
        '''
        if self.get_anno_type(anno_name) != "mask":
            raise ValueError(
                f"Annotation '{anno_name}' is not a mask. Type={self.get_anno_type(anno_name)}"
            )
        return self._anno_arr_dict[anno_name]

    def get_arr_anno(self, anno_name):
        '''
        Return array annotation with shape (N, ...).
        '''
        if self.get_anno_type(anno_name) != "array":
            raise ValueError(
                f"Annotation '{anno_name}' is not an array annotation. Type={self.get_anno_type(anno_name)}"
            )
        return self._anno_arr_dict[anno_name]

    def get_region_track_by_index(self, anno_name, index):
        if self.get_anno_type(anno_name) != "track":
            raise ValueError(
                f"Annotation '{anno_name}' is not a track. Type={self.get_anno_type(anno_name)}"
            )
        anno_arr = self._anno_arr_dict[anno_name]
        region_len = int(self.get_region_lens()[index])
        return anno_arr[index, :region_len]

    def get_region_stat_by_index(self, anno_name, index):
        if self.get_anno_type(anno_name) != "stat":
            raise ValueError(
                f"Annotation '{anno_name}' is not a stat. Type={self.get_anno_type(anno_name)}"
            )
        return self._anno_arr_dict[anno_name][index, 0]

    def get_region_mask_by_index(self, anno_name, index):
        if self.get_anno_type(anno_name) != "mask":
            raise ValueError(
                f"Annotation '{anno_name}' is not a mask. Type={self.get_anno_type(anno_name)}"
            )
        return self._anno_arr_dict[anno_name][index, 0]

    def get_region_array_by_index(self, anno_name, index):
        if self.get_anno_type(anno_name) != "array":
            raise ValueError(
                f"Annotation '{anno_name}' is not an array annotation. Type={self.get_anno_type(anno_name)}"
            )
        return self._anno_arr_dict[anno_name][index]
    
    def save_anno_npy(self, anno_name, npy_path):
        '''
        Save annotation to a numpy file.

        Keyword arguments:
        - anno_name: Name of the annotation.
        - npy_path: Path to save the annotation.

        Returns:
        None
        '''
        if anno_name not in self._anno_arr_dict:
            raise ValueError(f"Annotation '{anno_name}' not found. Available annotations: {list(self._anno_arr_dict.keys())}")
        np.save(npy_path, self._anno_arr_dict[anno_name])
    
    def save_anno_npz(self, anno_name, npz_path):
        '''
        Save annotation to a npz file.
        '''
        if anno_name not in self._anno_arr_dict:
            raise ValueError(f"Annotation '{anno_name}' not found. Available annotations: {list(self._anno_arr_dict.keys())}")
        np.savez_compressed(npz_path, self._anno_arr_dict[anno_name])

    @staticmethod
    def one_hot_encoding(seq: str):
        '''
        Return the one hot encoding of a sequence.
        Ambiguous nucleotides are encoded as zeros.
        Adapted from PROcapNet code.

        Keyword arguments:
        - seq: Sequence to encode.

        Output:
        - encoding: numpy array of shape (len(seq), 4)
        '''
        ambiguous_nucs = ["Y", "R", "W", "S", "K", "M", "D", "V", "H", "B", "X", "N"]

        sequence = seq.upper()
        if isinstance(sequence, str):
            sequence = list(sequence)

        alphabet = ["A", "C", "G", "T"]
        alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

        ohe = np.zeros((len(sequence), len(alphabet)), dtype="int8")
        for i, char in enumerate(sequence):
            if char in alphabet:
                idx = alphabet_lookup[char]
                ohe[i, idx] = 1
            else:
                assert char in ambiguous_nucs, char

        return ohe
