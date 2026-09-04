
import json
import os
from pathlib import Path

import numpy as np

from .BedTable import BedTable3, BedTable6Plus, BedTable3Plus
from .ExogenousSequences import ExogenousSequences
from .GeneralElements import GeneralElements
from .utils import reverse_complement_iupac, validate_iupac_dna

_REGION_SCHEMA_ROOT_FIELDS = frozenset({"schema_version", "base_type", "extra_columns"})
_REGION_SCHEMA_EXTRA_FIELDS = frozenset({"name", "dtype"})
_REGION_SCHEMA_DTYPE_NAMES = {
    "str": str,
    "int": int,
    "float": float,
}
_BED3_BASE_COLUMNS = ("chrom", "start", "end")
_BED6_BASE_COLUMNS = ("chrom", "start", "end", "name", "score", "strand")
_FORBIDDEN_NAME_CHARS = ("\t", "\n", "\r", "\0")


class GenomicElements(GeneralElements):
    '''
    Class for Genomics elements.
    '''

    def __init__(self, region_file_path, region_file_type, fasta_path):
        '''
        Constructor for GenomicElements class.

        Keyword arguments:
        - region_file_path: Path to the region file.
        - region_file_type: Schema selector — a predefined named region format,
          or a path to a version-1 region-schema JSON file.
        - fasta_path: Path to the genome file.
        '''
        schema_factory, schema_identity = self._load_region_schema(region_file_type)
        self._init_from_resolved_schema(
            region_file_path=region_file_path,
            region_file_type=region_file_type,
            fasta_path=fasta_path,
            schema_factory=schema_factory,
            schema_identity=schema_identity,
        )

    def _init_from_resolved_schema(
        self,
        *,
        region_file_path,
        region_file_type,
        fasta_path,
        schema_factory,
        schema_identity,
    ):
        super().__init__()
        self._region_file_path = region_file_path
        self._region_file_type = region_file_type
        self._region_schema_factory = schema_factory
        self._region_schema_identity = schema_identity
        self._region_bt = schema_factory(enable_sort=False)
        self._region_bt.load_from_file(self.region_file_path)
        self._fasta_path = fasta_path

    @classmethod
    def _from_resolved_schema(
        cls,
        *,
        region_file_path,
        region_file_type,
        fasta_path,
        schema_factory,
        schema_identity,
    ):
        '''Construct a collection using an already-resolved schema snapshot.'''
        obj = cls.__new__(cls)
        obj._init_from_resolved_schema(
            region_file_path=region_file_path,
            region_file_type=region_file_type,
            fasta_path=fasta_path,
            schema_factory=schema_factory,
            schema_identity=schema_identity,
        )
        return obj

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
        Return the dictionary that maps predefined named region formats to
        table constructors. Every named format constructs a BedTable3Plus or
        BedTable6Plus instance (including schemas with no extra columns).
        '''
        return {
            "bed3": GenomicElements.BedTable3Plain,
            "bed6": GenomicElements.BedTable6Plain,
            "bed6gene": GenomicElements.BedTable6Gene,
            "bed3gene": GenomicElements.BedTable3Gene,
            "narrowPeak": GenomicElements.BedTableNarrowPeak,
            "TREbed": GenomicElements.BedTableTREBed,
            "bedGraph": GenomicElements.BedTableBedGraph,
        }

    @staticmethod
    def BedTable3Plain(enable_sort=True):
        '''Return a BedTable3Plus with no extra columns (plain BED3 schema).'''
        return BedTable3Plus(
            extra_column_names=[],
            extra_column_dtype=[],
            enable_sort=enable_sort,
        )

    @staticmethod
    def BedTable6Plain(enable_sort=True):
        '''Return a BedTable6Plus with no extra columns (plain BED6 schema).'''
        return BedTable6Plus(
            extra_column_names=[],
            extra_column_dtype=[],
            enable_sort=enable_sort,
        )

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
    def _load_region_schema(selector):
        '''
        Resolve a schema selector to a table constructor and identity token.

        Predefined named formats take precedence over same-named files.
        Any other selector is treated as a schema-file path resolved against
        the process current working directory.
        '''
        predefined = GenomicElements.get_region_file_suffix2class_dict()
        if selector in predefined:
            return predefined[selector], ("named", selector)

        schema_path = Path(selector)
        if not schema_path.is_file():
            raise ValueError(
                f"Region schema selector {selector!r} is neither a supported "
                f"named format nor a readable schema file."
            )

        try:
            with schema_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Region schema file {selector!r} contains malformed JSON: {exc}."
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Region schema selector {selector!r} is neither a supported "
                f"named format nor a readable schema file."
            ) from exc

        extra_names, extra_dtypes, base_type = GenomicElements._parse_region_schema_payload(
            payload,
            source=selector,
        )
        table_cls = BedTable3Plus if base_type == "bed3" else BedTable6Plus
        captured_names = list(extra_names)
        captured_dtypes = list(extra_dtypes)

        def factory(enable_sort=True):
            return table_cls(
                extra_column_names=list(captured_names),
                extra_column_dtype=list(captured_dtypes),
                enable_sort=enable_sort,
            )

        identity = ("custom", os.path.realpath(str(schema_path)))
        return factory, identity

    @staticmethod
    def _parse_region_schema_payload(payload, *, source):
        if not isinstance(payload, dict):
            raise ValueError(
                f"Region schema {source!r} must be a JSON object at the root."
            )

        unexpected = sorted(set(payload) - _REGION_SCHEMA_ROOT_FIELDS)
        missing = sorted(_REGION_SCHEMA_ROOT_FIELDS - set(payload))
        if unexpected:
            raise ValueError(
                f"Region schema {source!r} has unsupported root field(s): "
                f"{', '.join(unexpected)}."
            )
        if missing:
            raise ValueError(
                f"Region schema {source!r} is missing required field(s): "
                f"{', '.join(missing)}."
            )

        schema_version = payload["schema_version"]
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError(
                f"Region schema {source!r} has unsupported schema_version "
                f"{schema_version!r}; only integer 1 is accepted."
            )

        base_type = payload["base_type"]
        if base_type not in ("bed3", "bed6"):
            raise ValueError(
                f"Region schema {source!r} has unsupported base_type "
                f"{base_type!r}; expected 'bed3' or 'bed6'."
            )

        extra_columns = payload["extra_columns"]
        if not isinstance(extra_columns, list):
            raise ValueError(
                f"Region schema {source!r} field extra_columns must be a JSON array."
            )

        base_columns = _BED3_BASE_COLUMNS if base_type == "bed3" else _BED6_BASE_COLUMNS
        extra_names = []
        extra_dtypes = []
        seen = set()

        for index, entry in enumerate(extra_columns):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Region schema {source!r} extra_columns[{index}] must be an object."
                )
            unexpected_extra = sorted(set(entry) - _REGION_SCHEMA_EXTRA_FIELDS)
            missing_extra = sorted(_REGION_SCHEMA_EXTRA_FIELDS - set(entry))
            if unexpected_extra:
                raise ValueError(
                    f"Region schema {source!r} extra_columns[{index}] has unsupported "
                    f"field(s): {', '.join(unexpected_extra)}."
                )
            if missing_extra:
                raise ValueError(
                    f"Region schema {source!r} extra_columns[{index}] is missing "
                    f"field(s): {', '.join(missing_extra)}."
                )

            name = entry["name"]
            dtype_name = entry["dtype"]
            if not isinstance(name, str) or name == "":
                raise ValueError(
                    f"Region schema {source!r} extra_columns[{index}] has an invalid "
                    f"name {name!r}; names must be nonempty strings."
                )
            if any(ch in name for ch in _FORBIDDEN_NAME_CHARS):
                raise ValueError(
                    f"Region schema {source!r} extra_columns[{index}] name {name!r} "
                    f"contains a forbidden control character."
                )
            if name in seen:
                raise ValueError(
                    f"Region schema {source!r} declares duplicate extra-column "
                    f"name {name!r}."
                )
            if name in base_columns:
                raise ValueError(
                    f"Region schema {source!r} extra-column name {name!r} collides "
                    f"with a {base_type} base column."
                )
            if dtype_name not in _REGION_SCHEMA_DTYPE_NAMES:
                raise ValueError(
                    f"Region schema {source!r} extra_columns[{index}] has unsupported "
                    f"dtype {dtype_name!r}; expected one of str, int, float."
                )

            seen.add(name)
            extra_names.append(name)
            extra_dtypes.append(_REGION_SCHEMA_DTYPE_NAMES[dtype_name])

        return extra_names, extra_dtypes, base_type

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

    def export_exogenous_sequences(
        self,
        fasta_path,
        *,
        output_orientation="genomic",
        record_id="coordinate",
    ):
        '''
        Export regions as exogenous sequences.

        Keyword arguments:
        - fasta_path: Path to save the exogenous sequences.
        - output_orientation: 'genomic' (default) or 'strand'.
        - record_id: 'coordinate' (default) or 'name'.

        Returns:
        - None
        '''
        if output_orientation not in ("genomic", "strand"):
            raise ValueError(
                "output_orientation must be 'genomic' or 'strand', "
                f"got {output_orientation!r}."
            )
        if record_id not in ("coordinate", "name"):
            raise ValueError(
                "record_id must be 'coordinate' or 'name', "
                f"got {record_id!r}."
            )
        if os.path.exists(fasta_path):
            raise ValueError(f"File {fasta_path} already exists.")

        region_bt = self.get_region_bed_table()
        fields = set(region_bt.column_names)
        has_strand = "strand" in fields
        has_name = "name" in fields

        if output_orientation == "strand" and not has_strand:
            raise ValueError(
                "output_orientation 'strand' requires a region schema with "
                "a row-level strand field."
            )
        if record_id == "name" and not has_name:
            raise ValueError(
                "record_id 'name' requires a region schema with "
                "a row-level name field."
            )

        genome_index = self._get_genome_index()
        regions = list(region_bt.iter_regions())
        for region in regions:
            chrom = region["chrom"]
            start = region["start"]
            end = region["end"]
            locus = f"{chrom}:{start}-{end}"
            if chrom not in genome_index:
                raise ValueError(
                    f"Chromosome {chrom} not found in genome file. "
                    f"Cannot export region {locus}"
                )
            chrom_len = len(genome_index[chrom].seq)
            if not (0 <= start < end <= chrom_len):
                raise ValueError(
                    f"Interval {locus} is not fully contained in "
                    f"chromosome {chrom} of length {chrom_len}."
                )

        seq_ids = []
        sequences = []
        seen_ids = set()

        for region in regions:
            chrom = region["chrom"]
            start = region["start"]
            end = region["end"]
            locus = f"{chrom}:{start}-{end}"
            seq = str(genome_index[chrom].seq[start:end])

            if output_orientation == "strand":
                strand = region["strand"]
                if strand not in ("+", "-"):
                    raise ValueError(
                        "output_orientation 'strand' requires strand '+' or '-', "
                        f"got {strand!r} for region {locus}."
                    )
                validate_iupac_dna(seq)
                if strand == "-":
                    seq = reverse_complement_iupac(seq)

            if record_id == "coordinate":
                record = locus
            else:
                name = region["name"]
                if name is None or (isinstance(name, float) and np.isnan(name)):
                    raise ValueError(
                        f"record_id 'name' requires a nonempty name for "
                        f"region {locus}."
                    )
                record = str(name)
                if record == "" or record == "." or any(ch.isspace() for ch in record):
                    raise ValueError(
                        f"Invalid record name {record!r} for region {locus}."
                    )

            if record in seen_ids:
                raise ValueError(
                    f"Duplicate FASTA record ID {record!r} under "
                    f"record_id={record_id!r}."
                )
            seen_ids.add(record)
            seq_ids.append(record)
            sequences.append(seq)

        ExogenousSequences.write_sequences_to_fasta(seq_ids, sequences, fasta_path)

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

        result_ge = self.__class__._from_resolved_schema(
            region_file_path=new_region_file_path,
            region_file_type=self.region_file_type,
            fasta_path=self.fasta_path,
            schema_factory=self._region_schema_factory,
            schema_identity=self._region_schema_identity,
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
    