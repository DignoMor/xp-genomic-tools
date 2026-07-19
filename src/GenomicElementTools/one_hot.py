
import numpy as np

from RGTools.GenomicElements import GenomicElements

class OneHot:
    @staticmethod
    def main(args):
        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=args.fasta_path, 
                                           )

        output_arr = genomic_elements.get_all_region_one_hot()
        output_arr = output_arr.transpose(0, 2, 1)
        np.save(args.opath, output_arr)

    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genome(parser)
        GenomicElements.set_parser_genomic_element_region(parser)

        parser.add_argument("--opath",
                            help="Output path for the one-hot encoded sequence.",
                            type=str,
                            required=True,
                            )
