import numpy as np
from RGTools.GenomicElements import GenomicElements

class FilterMotifScore:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)

        parser.add_argument("--motif_search_npy",
                            help="Numpy file containing motif search scores.",
                            required=True,
                            )

        parser.add_argument("--output_header",
                            help="Output header for filtered GenomicElements bed file.",
                            required=True,
                            )
    
        parser.add_argument("--filter_base", 
                            help="Base index for filtering motif search scores.",
                            type=int,
                            required=True,
                            )
        
        parser.add_argument("--min_score",
                            help="Minimum score for filtering motif search scores.",
                            type=float,
                            default=-np.inf,
                            )

        parser.add_argument("--max_score", 
                            help="Maximum score for filtering motif search scores.",
                            type=float,
                            default=np.inf,
                            )
        
    @staticmethod
    def main(args):
        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=None, 
                                           )
        genomic_elements.load_region_anno_from_npy("motif", args.motif_search_npy, anno_type="track")

        motif_track_list = genomic_elements.get_track_list("motif")
        filter_base_score = np.array([m[args.filter_base] for m in motif_track_list])
        filter_mask = (filter_base_score > args.min_score) & (filter_base_score < args.max_score)

        output_ge = genomic_elements.apply_logical_filter(filter_mask, 
                                                          args.output_header + ".bed",
                                                          )
                         
        output_ge.save_anno_npy("motif", 
                                args.output_header + ".motif.npy",
                                )

