
import os

import pandas as pd

from Bio import SeqIO

from .GeneralElements import GeneralElements
from .BedTable import BedTable3

class ExogeneousSequences(GeneralElements):
    '''
    Class for Exogeneous sequences.

    This class inherits from GeneralElements and provides
    functionality for handling exogeneous sequences (sequences
    that are not part of a reference genome).

    Usually exogeneous sequences are small sets 
    of sequences compared to reference genome. 
    As a result this class read sequences into 
    memory for further analysis.
    '''
    def __init__(self, fasta_path):
        '''
        Initialize the ExogeneousSequences object.

        Key arguments:
            - fasta_path: path to the fasta file
        '''
        super().__init__()
        self._fasta_path = fasta_path
        self._region_file_type = "bed3"

        self.sequence_df = self.read_fasta_sequences(fasta_path)
    
    @property
    def fasta_path(self):
        return self._fasta_path

    @property
    def region_file_type(self):
        return self._region_file_type

    @property
    def region_file_path(self):
        raise NotImplementedError("ExogeneousSequences does not use region_file_path. ")
    
    def get_region_bed_table(self):
        '''
        Return a bed table object for the sequences.
        Creates a BedTable3 with each sequence as a separate region.
        '''
        bt = BedTable3(enable_sort=False)
        
        # Create a dataframe with each sequence as a region
        # Use sequence name as chromosome, start=0, end=sequence length
        region_df = pd.DataFrame({
            "chrom": self.sequence_df.index,
            "start": 0,
            "end": [len(seq) for seq in self.sequence_df["seqs"]], 
        })
        
        bt.load_from_dataframe(region_df)
        return bt
    
    def read_fasta_sequences(self, fasta_path):
        '''
        Read fasta sequences and return a dataframe.
        '''

        seq_names = []
        seqs = []

        with open(fasta_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_names.append(str(record.id))
                seqs.append(str(record.seq))
        
        seq_lens = [len(seq) for seq in seqs]
        
        out_df = pd.DataFrame({"seqs": seqs,
                               "lens": seq_lens,
                              }, 
                              index=seq_names,
                              )
        
        return out_df

    def get_sequence_ids(self):
        '''
        Return an array of sequence ids.
        '''
        return self.get_region_bed_table().get_chrom_names()
    
    def get_all_region_seqs(self):
        return self.sequence_df["seqs"].tolist()
 
    def get_all_region_lens(self):
        return self.sequence_df["lens"].tolist()
    
    def apply_logical_filter(self, logical, new_fasta_path):
        '''
        Apply logical filter to the sequences.

        Keyword arguments:
        - logical: np.Array, Logical array to filter the sequences.
        - new_fasta_path: Path to save the new fasta file for filtered sequences.

        Returns:
        - a new ExogeneousSequences object with the filtered sequences.
        '''
        if os.path.exists(new_fasta_path):
            raise ValueError(f"File {new_fasta_path} already exists.")
        
        # Filter sequences based on logical array
        filtered_df = self.sequence_df[logical]
        
        # Write filtered sequences to new fasta file
        with open(new_fasta_path, "w") as f:
            for seq_id, row in filtered_df.iterrows():
                f.write(f">{seq_id}\n{row['seqs']}\n")
        
        # Create new ExogeneousSequences object
        result_es = self.__class__(fasta_path=new_fasta_path)
        
        # Copy filtered annotations
        for anno_name, anno_arr in self._anno_arr_dict.items():
            new_anno_arr = anno_arr[logical]
            anno_type = self.get_anno_type(anno_name)
            if anno_type == "track":
                new_max_len = int(result_es.get_region_lens().max())
                if new_anno_arr.shape[1] != new_max_len:
                    new_anno_arr = new_anno_arr[:, :new_max_len]
                new_region_lens = result_es.get_region_lens()
                track_list = [new_anno_arr[i, :new_region_lens[i]] for i in range(new_anno_arr.shape[0])]
                result_es.load_region_track_from_list(anno_name, track_list)
            elif anno_type == "stat":
                result_es.load_region_stat_from_arr(anno_name, new_anno_arr)
            elif anno_type == "mask":
                result_es.load_mask_from_arr(anno_name, new_anno_arr)
            elif anno_type == "array":
                result_es.load_region_array_from_arr(anno_name, new_anno_arr)
            else:
                raise ValueError(f"Invalid annotation type: {anno_type}")

        return result_es
    
    @staticmethod
    def set_parser_genome(parser):
        parser.add_argument("--fasta", 
                            help="Path to the sequence fasta file.",
                            required=True,
                            )

    @staticmethod
    def set_parser_exogeneous_sequences(parser):
        ExogeneousSequences.set_parser_genome(parser)
    
    @staticmethod
    def write_sequences_to_fasta(seq_ids: list, sequences: list, fasta_path: str):
        '''
        Write sequences to a fasta file.

        Key arguments:
            - seq_ids: list of sequence ids
            - sequences: list of sequences
            - fasta_path: path to the fasta file
        '''
        if os.path.exists(fasta_path):
            raise ValueError(f"File {fasta_path} already exists.")
        
        with open(fasta_path, "w") as f:
            for seq_id, seq in zip(seq_ids, sequences):
                f.write(f">{seq_id}\n{seq}\n")


