
import os 

import numpy as np

from .BedTable import BedTable3, BedTable6, BedTable6Plus, BedTable3Plus
from .GeneralElements import GeneralElements

class GenomicElements(GeneralElements):
    '''
    Class for Genomics elements.
    '''

    def __init__(self, region_file_path, region_file_type, fasta_path):
        '''
        Constructor for GenomicElements class.

        Keyword arguments:
        - region_file_path: Path to the region file.
        - region_file_type: Type of the region file (see GenomicElements.get_region_file_suffix2class_dict).
        - fasta_path: Path to the genome file.
        '''
        super().__init__()
        self._region_file_path = region_file_path
        self._region_file_type = region_file_type
        if not self._region_file_type in self.get_region_file_suffix2class_dict().keys():
            raise ValueError(f"Invalid region file type: {self._region_file_type}")

        self._region_bt = self.get_region_file_suffix2class_dict()[self.region_file_type](enable_sort=False)
        self._region_bt.load_from_file(self.region_file_path)

        self._fasta_path = fasta_path

    @property
    def fasta_path(self):
        return self._fasta_path

    @property
    def region_file_type(self):
        return self._region_file_type

    @property
    def region_file_path(self):
        return self._region_file_path

    def get_num_regions(self):
        '''
        Return the number of regions without copying the bed table.
        '''
        return len(self._region_bt)

    @staticmethod
    def get_region_file_suffix2class_dict():
        '''
        Return the dictionary that maps region file suffix to the corresponding class.
        '''
        return {
            "bed3": BedTable3,
            "bed6": BedTable6,
            "bed6gene": GenomicElements.BedTable6Gene,
            "bed3gene": GenomicElements.BedTable3Gene,
            "narrowPeak": GenomicElements.BedTableNarrowPeak,
            "TREbed": GenomicElements.BedTableTREBed,
            "bedGraph": GenomicElements.BedTableBedGraph,
        }

    @staticmethod
    def BedTable6Gene(enable_sort=True):
        '''
        Helper function to return a BedTable6Plus object
        that can load bed6gene annotations.
        '''
        bt = BedTable6Plus(extra_column_names=["gene_symbol"], 
                           extra_column_dtype=[str], 
                           enable_sort=enable_sort,
                           )
        return bt

    @staticmethod
    def BedTable3Gene(enable_sort=True):
        '''
        Helper function to return a BedTable3Plus object
        that can load bed3gene annotations.
        '''
        bt = BedTable3Plus(extra_column_names=["gene_symbol"], 
                           extra_column_dtype=[str], 
                           enable_sort=enable_sort,
                           )
        return bt
    
    @staticmethod
    def BedTableNarrowPeak(enable_sort=True):
        bt = BedTable6Plus(extra_column_names=["signalValue", "pValue", "qValue", "peak"], 
                           extra_column_dtype=[float, float, float, int], 
                           enable_sort=enable_sort,
                           )
        return bt
    
    @staticmethod
    def BedTableBedGraph(enable_sort=True):
        bt = BedTable3Plus(extra_column_names=["dataValue"], 
                           extra_column_dtype=[float], 
                           enable_sort=enable_sort,
                           )
        return bt

    @staticmethod
    def BedTableTREBed(enable_sort=True):
        bt = BedTable3Plus(extra_column_names=["name", "fwdTSS", "revTSS"], 
                           extra_column_dtype=[str, int, int], 
                           enable_sort=enable_sort,
                           )
        return bt

    @staticmethod
    def set_parser_genome(parser):
        parser.add_argument("--fasta_path", 
                            help="Path to the genome file.",
                            required=True,
                            type=str, 
                            )

    @staticmethod
    def set_parser_genomic_element_region(parser):
        parser.add_argument("--region_file_path", 
                            help="Path to the region file.",
                            required=True,
                            type=str, 
                            )
        
        parser.add_argument("--region_file_type",
                            help="Type of the region file. "
                                 "Valid types: {}".format(
                                     list(GenomicElements.get_region_file_suffix2class_dict().keys())
                                     ),
                            required=True,
                            default="bed3", 
                            type=str, 
                            choices=GenomicElements.get_region_file_suffix2class_dict().keys(),
                            )

    @staticmethod
    def merge_genomic_elements(left_ge, right_ge, output_region_path, anno2merge, sort_new_ge=True):
        '''
        Merge two GenomicElements objects and write merged regions to disk.

        Keyword arguments:
        - left_ge: Left GenomicElements object.
        - right_ge: Right GenomicElements object.
        - output_region_path: Path to write merged regions.
        - anno2merge: list of annotation names to merge from both objects.
        - sort_new_ge: Whether to sort merged regions before writing.

        Returns:
        - A new GenomicElements object loaded from output_region_path.
        '''
        if not isinstance(left_ge, GenomicElements) or not isinstance(right_ge, GenomicElements):
            raise ValueError("left_ge and right_ge must both be GenomicElements instances.")

        if left_ge.region_file_type != right_ge.region_file_type:
            raise ValueError(
                f"Cannot merge GenomicElements with different region_file_type: "
                f"{left_ge.region_file_type} vs {right_ge.region_file_type}"
            )

        if left_ge.fasta_path != right_ge.fasta_path:
            raise ValueError(
                f"Cannot merge GenomicElements with different fasta_path: "
                f"{left_ge.fasta_path} vs {right_ge.fasta_path}"
            )

        left_bt = left_ge.get_region_bed_table()
        right_bt = right_ge.get_region_bed_table()
        if sort_new_ge:
            new_bt, merge_index_map = BedTable3._merge_sort_bt(left_bt, right_bt, return_index=True)
        else:
            merged_regions = list(left_bt.iter_regions()) + list(right_bt.iter_regions())
            new_bt = left_bt._clone_empty()
            new_bt.load_from_bed_regions(merged_regions)
            left_n = len(left_bt)
            right_n = len(right_bt)
            left_map = np.column_stack([np.zeros(left_n, dtype=int), np.arange(left_n, dtype=int)])
            right_map = np.column_stack([np.ones(right_n, dtype=int), np.arange(right_n, dtype=int)])
            merge_index_map = np.concatenate([left_map, right_map], axis=0)

        new_bt.write(output_region_path)

        result_ge = GenomicElements(output_region_path,
                                    left_ge.region_file_type,
                                    left_ge.fasta_path,
                                    )

        for anno_name in anno2merge:
            if anno_name not in left_ge._anno_arr_dict or anno_name not in right_ge._anno_arr_dict:
                raise ValueError(f"Annotation '{anno_name}' must exist in both input GenomicElements.")

            left_type = left_ge.get_anno_type(anno_name)
            right_type = right_ge.get_anno_type(anno_name)
            if left_type != right_type:
                raise ValueError(
                    f"Annotation '{anno_name}' type mismatch: left={left_type}, right={right_type}"
                )

            if left_type in ("stat", "mask"):
                if left_type == "stat":
                    left_arr = left_ge.get_stat_arr(anno_name)
                    right_arr = right_ge.get_stat_arr(anno_name)
                else:
                    left_arr = left_ge.get_mask_arr(anno_name)
                    right_arr = right_ge.get_mask_arr(anno_name)
                if left_arr.shape[1] != right_arr.shape[1]:
                    raise ValueError(
                        f"Annotation '{anno_name}' dim mismatch: left={left_arr.shape[1]}, right={right_arr.shape[1]}"
                    )
                merged_list = []
                for source_table, source_index in merge_index_map:
                    if source_table == 0:
                        merged_list.append(left_arr[source_index])
                    else:
                        merged_list.append(right_arr[source_index])
            elif left_type == "track":
                left_list = left_ge.get_track_list(anno_name)
                right_list = right_ge.get_track_list(anno_name)
                merged_list = []
                for source_table, source_index in merge_index_map:
                    track = left_list[source_index] if source_table == 0 else right_list[source_index]
                    merged_list.append(track)
            elif left_type == "array":
                left_arr = left_ge.get_arr_anno(anno_name)
                right_arr = right_ge.get_arr_anno(anno_name)
                if left_arr.shape[1:] != right_arr.shape[1:]:
                    raise ValueError(
                        f"Annotation '{anno_name}' array shape mismatch: "
                        f"left={left_arr.shape[1:]}, right={right_arr.shape[1:]}"
                    )
                merged_list = []
                for source_table, source_index in merge_index_map:
                    if source_table == 0:
                        merged_list.append(left_arr[source_index])
                    else:
                        merged_list.append(right_arr[source_index])
            else:
                raise ValueError(f"Unsupported annotation type: {left_type}")

            if left_type == "track":
                result_ge.load_region_track_from_list(anno_name, merged_list)
            elif left_type == "stat":
                result_ge.load_region_stat_from_arr(anno_name, np.asarray(merged_list))
            elif left_type == "mask":
                result_ge.load_mask_from_arr(anno_name, np.asarray(merged_list))
            elif left_type == "array":
                result_ge.load_region_array_from_arr(anno_name, np.asarray(merged_list))

        return result_ge

    def export_exogeneous_sequences(self, fasta_path):
        '''
        Export regions as exogeneous sequences.

        Keyword arguments:
        - fasta_path: Path to save the exogeneous sequences.

        Returns:
        - None
        '''
        if os.path.exists(fasta_path):
            raise ValueError(f"File {fasta_path} already exists.")
        
        for region in self.get_region_bed_table().iter_regions():
            seq = self.get_region_seq(region["chrom"], region["start"], region["end"])
            if seq is None:
                raise ValueError(f"Chromosome {region['chrom']} not found in genome file. Cannot export region {region['chrom']}:{region['start']}-{region['end']}")
            with open(fasta_path, "a") as handle:
                handle.write(f">{region['chrom']}:{region['start']}-{region['end']}\n{seq}\n")

    def get_all_region_seqs(self):
        '''
        Return sequences for all regions in the region table.
        '''
        out_seqs = []
        for region in self.get_region_bed_table().iter_regions():
            seq = self.get_region_seq(region["chrom"], region["start"], region["end"], index_genome=True)
            if seq is None:
                raise ValueError(f"Chromosome {region['chrom']} not found in the genome file.")
            out_seqs.append(seq)
        return out_seqs

    def get_region_bed_table(self):
        '''
        Return a bed table object for the region file.
        '''
        return self._region_bt.copy()
    
    def apply_logical_filter(self, logical, new_region_file_path):
        '''
        Apply logical filter to the regions.

        Keyword arguments:
        - logical: np.Array, Logical array to filter the regions.
        - new_region_file_path: Path to save the new region_file for filtered regions.

        Returns:
        - a new GenomicElements object with the filtered regions.
        '''
        logical = np.asarray(logical)
        if logical.dtype != np.bool_:
            raise ValueError(f"Logical array must have boolean dtype; got dtype {logical.dtype}")
        if logical.ndim != 1:
            raise ValueError(f"Logical array must be 1D; got shape {logical.shape}")
        num_regions = self.get_num_regions()
        if len(logical) != num_regions:
            raise ValueError(f"Logical array length ({len(logical)}) does not match number of regions ({num_regions})")
        
        result_bt = self.get_region_bed_table().apply_logical_filter(logical)
        result_bt.write(new_region_file_path)

        result_ge = self.__class__(region_file_path=new_region_file_path,
                                   region_file_type=self.region_file_type,
                                   fasta_path=self.fasta_path,
                                   )
        
        for anno_name, anno_arr in self._anno_arr_dict.items():
            new_anno_arr = anno_arr[logical]
            anno_type = self.get_anno_type(anno_name)
            if anno_type == "track":
                new_max_len = int(result_ge.get_region_lens().max())
                if new_anno_arr.shape[1] != new_max_len:
                    new_anno_arr = new_anno_arr[:, :new_max_len]
                new_region_lens = result_ge.get_region_lens()
                track_list = [new_anno_arr[i, :new_region_lens[i]] for i in range(new_anno_arr.shape[0])]
                result_ge.load_region_track_from_list(anno_name, track_list)
            elif anno_type == "stat":
                result_ge.load_region_stat_from_arr(anno_name, new_anno_arr)
            elif anno_type == "mask":
                result_ge.load_mask_from_arr(anno_name, new_anno_arr)
            elif anno_type == "array":
                result_ge.load_region_array_from_arr(anno_name, new_anno_arr)
            else:
                raise ValueError(f"Invalid annotation type: {anno_type}")

        return result_ge
    