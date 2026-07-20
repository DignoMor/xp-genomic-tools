

import numpy as np

from RGTools.GenomicElements import GenomicElements
from RGTools.BwTrack import SingleBwTrack, PairedBwTrack
from RGTools.utils import str2bool

class CountSingleBw:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--bw_path",
                            help="Bigwig file path.",
                            required=True,
                            type=str,
                            )

        parser.add_argument("--quantification_type",
                            help="Type of quantification.",
                            type=str,
                            default="raw_count",
                            choices=SingleBwTrack.get_supported_quantification_type(),
                            )

        parser.add_argument("--opath",
                            help="Output path for counting.",
                            required=True,
                            type=str,
                            )

    @staticmethod
    def main(args):
        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=None, 
                                           )
        region_bt = genomic_elements.get_region_bed_table()

        bw_track = SingleBwTrack(bw_path=args.bw_path)

        output_list = []
        for region in region_bt.iter_regions():

            output_list.append(bw_track.count_single_region(region["chrom"],
                                                            region["start"],
                                                            region["end"],
                                                            output_type=args.quantification_type,
                                                            min_len_after_padding=1, 
                                                            ),
                               )

        if args.quantification_type == "full_track":
            genomic_elements.load_region_track_from_list("count", output_list)
        else:
            genomic_elements.load_region_stat_from_arr("count", np.array(output_list))

        if args.opath.endswith(".npz"):
            genomic_elements.save_anno_npz("count", args.opath)
        else:
            genomic_elements.save_anno_npy("count", args.opath)

class CountPairedBw:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--bw_pl",
                            help="Plus strand bigwig file.",
                            required=True,
                            type=str,
                            )

        parser.add_argument("--bw_mn",
                            help="Minus strand bigwig file.",
                            required=True,
                            type=str,
                            )

        parser.add_argument("--override_strand",
                            help="Override the strand information in the input file (None if use the input strand info).",
                            type=str,
                            default=None,
                            )

        parser.add_argument("--quantification_type",
                            help="Type of quantification.",
                            type=str,
                            default="raw_count",
                            choices=PairedBwTrack.get_supported_quantification_type(),
                            )

        parser.add_argument("--negative_mn",
                            help="Whether to output the minus strand signal as negative.",
                            type=str2bool,
                            required=True,
                            )

        parser.add_argument("--flip_mn",
                            help="If to flip the minus strand signal.",
                            type=str2bool,
                            required=True,
                            )

        parser.add_argument("--opath",
                            help="Output path for counting.",
                            required=True,
                            type=str,
                            )

    @staticmethod
    def main(args):
        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=None, 
                                           )
        region_bt = genomic_elements.get_region_bed_table()

        bw_track = PairedBwTrack(bw_pl_path=args.bw_pl,
                                 bw_mn_path=args.bw_mn,
                                 )

        output_list = []
        for region in region_bt.iter_regions():
            if args.override_strand:
                strand = args.override_strand
            elif args.region_file_type == "bed3":
                strand = "."
            elif not region["strand"]:
                strand = "."
            else:
                strand = region["strand"]

            output_list.append(bw_track.count_single_region(region["chrom"],
                                                            region["start"],
                                                            region["end"],
                                                            strand, 
                                                            output_type=args.quantification_type,
                                                            min_len_after_padding=1, 
                                                            flip_mn=args.flip_mn,
                                                            negative_mn=args.negative_mn,
                                                            ),
                               )

        if args.quantification_type == "full_track":
            genomic_elements.load_region_track_from_list("count", output_list)
        else:
            genomic_elements.load_region_stat_from_arr("count", np.array(output_list))

        if args.opath.endswith(".npz"):
            genomic_elements.save_anno_npz("count", args.opath)
        else:
            genomic_elements.save_anno_npy("count", args.opath)
    
