
from RGTools.GenomicElements import GenomicElements
from RGTools.ListFile import ListFile
from RGTools.BedTable import BedTable3, BedRegion
from RGTools.ExogeneousSequences import ExogeneousSequences

import numpy as np
import re

class GenomicElementImport:
    @staticmethod
    def set_parser(parser):
        subparsers = parser.add_subparsers(dest="informat", 
                                           help="Input format. [{}]".format(", ".join(GenomicElementImport.get_informat_options())),
                                           required=True,
                                           )

        parser_stat_list = subparsers.add_parser(
            "stat_list",
            help="Import a list file as stat annotation values and output a GenomicElements npy/npz file.",
        )
        GenomicElementImport.set_parser_stat_list(parser_stat_list)

        parser_allele_expanded_es = subparsers.add_parser(
            "allele_expanded_ES",
            help="Import allele-expanded ExogeneousSequences FASTA and derive per-region outputs.",
        )
        GenomicElementImport.set_parser_allele_expanded_es(parser_allele_expanded_es)

    @staticmethod
    def set_parser_stat_list(parser):
        GenomicElements.set_parser_genomic_element_region(parser)

        parser.add_argument("--inpath", "-I", 
                            help="Input path of the list file containing one stat value per region.",
                            required=True,
                            )
        
        parser.add_argument("--opath", 
                            help="Output path of the region file.",
                            required=True,
                            )
        
        parser.add_argument("--dtype", 
                            help="Dtype of the outputted array.",
                            default="str",
                            type=str,
                            choices=["str", "np.int32", "np.int64", "np.float32", "np.float64"],
                            )

        return parser

    @staticmethod
    def set_parser_allele_expanded_es(parser):
        parser.add_argument("--inpath", "-I",
                            help="Input path of the allele-expanded FASTA file.",
                            required=True,
                            )

        parser.add_argument("--anno_oheader",
                            help="Output header for generated files.",
                            required=True,
                            )

        parser.add_argument("--stat_name",
                            help="Name of stat annotation to import (append once per stat file).",
                            action="append",
                            required=False,
                            default=[],
                            )

        parser.add_argument("--stat_npy",
                            help="Path to stat npy/npz file aligned to FASTA entries (append once per stat_name).",
                            action="append",
                            required=False,
                            default=[],
                            )

        parser.add_argument("--stat_selection_method",
                            help="Method to pick a representative alternate stat per region.",
                            action="append",
                            required=False,
                            default=[],
                            choices=["max_abs_fc"],
                            )
        return parser

    @staticmethod
    def get_informat_options():
        return ["stat_list", "allele_expanded_ES"]

    @staticmethod
    def _parse_allele_expanded_seq_id(seq_id):
        ref_match = re.match(r"^(.+)_(\d+)_(\d+)_ref$", seq_id)
        if ref_match:
            return {
                "chrom": ref_match.group(1),
                "start": int(ref_match.group(2)),
                "end": int(ref_match.group(3)),
                "is_ref": True,
                "snp_pos": None,
            }

        alt_match = re.match(r"^(.+)_(\d+)_(\d+)_(\d+):([A-Za-z])2([A-Za-z])$", seq_id)
        if alt_match:
            return {
                "chrom": alt_match.group(1),
                "start": int(alt_match.group(2)),
                "end": int(alt_match.group(3)),
                "is_ref": False,
                "snp_pos": int(alt_match.group(4)),
            }

        raise ValueError(
            f"Invalid allele_expanded_ES FASTA header '{seq_id}'. "
            "Expected 'chrom_start_end_ref' or 'chrom_start_end_<snp_position>:<ref_base>2<alt_base>'."
        )

    @staticmethod
    def _select_alt_stat(ref_stat, alt_stats, method):
        '''
        Used in `import_allele_expanded_es` to select the alt stat value based on the reference stat value and the method.
        '''
        if len(alt_stats) == 0:
            return np.nan
        if method == "max_abs_fc":
            alt_stats = np.asarray(alt_stats)
            return alt_stats[np.argmax(np.abs(alt_stats - ref_stat))]
        raise ValueError(f"Unsupported stat_selection_method: {method}")

    @staticmethod
    def import_allele_expanded_es(args):
        if not (len(args.stat_name) == len(args.stat_npy) == len(args.stat_selection_method)):
            raise ValueError(
                "Lengths of --stat_name, --stat_npy, and --stat_selection_method must match. "
                f"Got {len(args.stat_name)}, {len(args.stat_npy)}, and {len(args.stat_selection_method)}."
            )

        input_es = ExogeneousSequences(args.inpath)
        seq_ids = list(input_es.get_sequence_ids())
        if len(seq_ids) == 0:
            raise ValueError(f"No FASTA entries found in {args.inpath}")

        group_dict = {}
        ordered_keys = []
        for i, seq_id in enumerate(seq_ids):
            parsed_id = GenomicElementImport._parse_allele_expanded_seq_id(str(seq_id))
            key = (parsed_id["chrom"], parsed_id["start"], parsed_id["end"])
            if key not in group_dict:
                group_dict[key] = {"ref_index": None, "alt_indices": []}
                ordered_keys.append(key)

            if parsed_id["is_ref"]:
                if group_dict[key]["ref_index"] is not None:
                    raise ValueError(f"Duplicate reference FASTA entry for region {key}.")
                group_dict[key]["ref_index"] = i
            else:
                group_dict[key]["alt_indices"].append(i)

        missing_ref = [k for k, v in group_dict.items() if v["ref_index"] is None]
        if len(missing_ref) > 0:
            raise ValueError(f"Missing reference FASTA entry for regions: {missing_ref}")

        bed3_path = args.anno_oheader + ".bed3"
        output_bt = BedTable3(enable_sort=True)
        output_bt.load_from_bed_regions([
            BedRegion(chrom=k[0], start=k[1], end=k[2]) for k in ordered_keys
        ])
        output_bt.write(bed3_path)

        output_ge = GenomicElements(region_file_path=bed3_path,
                                    region_file_type="bed3",
                                    fasta_path=None,
                                    )
        num_fasta_records = len(seq_ids)
        for stat_name, stat_npy, stat_method in zip(args.stat_name, args.stat_npy, args.stat_selection_method):
            stat_arr = np.load(stat_npy, allow_pickle=False)
            if hasattr(stat_arr, "files"):
                keys = list(stat_arr.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ file {stat_npy} contains multiple arrays ({len(keys)}). "
                        "Please use a single-array npz or npy."
                    )
                stat_arr = stat_arr[keys[0]]

            stat_arr = np.asarray(stat_arr).reshape(-1,)
            if len(stat_arr) != num_fasta_records:
                raise ValueError(
                    f"Stat array length {len(stat_arr)} in {stat_npy} does not match "
                    f"number of FASTA entries {num_fasta_records} in {args.inpath}"
                )

            ref_stats = []
            alt_stats = []
            for key in ordered_keys:
                ref_index = group_dict[key]["ref_index"]
                alt_indices = group_dict[key]["alt_indices"]
                ref_val = stat_arr[ref_index]
                alt_val = GenomicElementImport._select_alt_stat(ref_val, stat_arr[alt_indices], stat_method)
                ref_stats.append(ref_val)
                alt_stats.append(alt_val)

            output_ge.load_region_stat_from_arr(f"{stat_name}.ref", np.asarray(ref_stats))
            output_ge.load_region_stat_from_arr(f"{stat_name}.alt", np.asarray(alt_stats))
            output_ge.save_anno_npy(f"{stat_name}.ref", f"{args.anno_oheader}.{stat_name}.ref.npy")
            output_ge.save_anno_npy(f"{stat_name}.alt", f"{args.anno_oheader}.{stat_name}.alt.npy")

    @staticmethod
    def import_stat_list(args):
        '''
        Import stat annotation values from a list file.
        '''
        input_ge = GenomicElements(region_file_path=args.region_file_path,
                                   region_file_type=args.region_file_type,
                                   fasta_path=None,
                                   )
        lsfile = ListFile()
        lsfile.read_file(args.inpath)
        
        if not input_ge.get_num_regions() == lsfile.get_num_lines():
            raise ValueError(f"Number of regions in the input file {lsfile.get_num_lines()} does not match the number of regions in the region file {input_ge.get_num_regions()}")

        region_arr = lsfile.get_contents(dtype=args.dtype)

        input_ge.load_region_stat_from_arr("region_list", region_arr)

        if args.opath.endswith(".npz"): 
            input_ge.save_anno_npz("region_list", args.opath)
        elif args.opath.endswith(".npy"):
            input_ge.save_anno_npy("region_list", args.opath)
        else:
            raise ValueError(f"Invalid output file type: {args.opath}")

    @staticmethod
    def main(args):
        if args.informat == "stat_list":
            GenomicElementImport.import_stat_list(args)
        elif args.informat == "allele_expanded_ES":
            GenomicElementImport.import_allele_expanded_es(args)
        else:
            raise ValueError(f"Invalid input format: {args.informat}")
