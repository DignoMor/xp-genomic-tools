
import pandas as pd

from RGTools.GenomicElements import GenomicElements
from RGTools.exceptions import InvalidBedRegionException
from RGTools.utils import str2bool

class PadRegion:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--upstream_pad",
                            help="Amount to extend to the upstream of the region. "
                                 "Positive value will expand the region and negative value will shrink the region.",
                            type=int,
                            required=True,
                            )

        parser.add_argument("--downstream_pad",
                            help="Amount to extend to the downstream of the region. "
                                 "Positive value will expand the region and negative value will shrink the region.",
                            type=int,
                            required=True,
                            )

        parser.add_argument("--opath",
                            help="Output path for the padded BED file",
                            type=str,
                            required=True,
                            )

        parser.add_argument("--ignore_strand",
                            help="Ignore the strand information in the input file. ",
                            default=False,
                            type=str2bool,
                            )
        
        parser.add_argument("--method_resolving_invalid_region", 
                            help="Method to resolve invalid region after padding.", 
                            type=str,
                            default="fallback",
                            choices=["raise", "fallback", "drop"],
                            )

    @staticmethod
    def main(args):
        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=None, 
                                           )
        
        region_bt = genomic_elements.get_region_bed_table()

        output_dict_list = []

        for i, region in enumerate(region_bt.iter_regions()):
            try:
                new_region = region.pad_region(upstream_padding=args.upstream_pad,
                                               downstream_padding=args.downstream_pad,
                                               ignore_strand=args.ignore_strand,
                                               )
            except InvalidBedRegionException as e:
                if args.method_resolving_invalid_region == "raise":
                    raise e
                elif args.method_resolving_invalid_region == "fallback":
                    new_region = region
                elif args.method_resolving_invalid_region == "drop":
                    continue
                else:
                    raise ValueError(f"Unknown method to resolve invalid region: {args.method_resolving_invalid_region}")

            output_dict_list.append(new_region.to_dict())
        
        output_region_bt = region_bt._clone_empty()
        output_region_bt.load_from_dataframe(pd.DataFrame(output_dict_list,
                                                          columns=region_bt.column_names,
                                                          ))
            
        output_region_bt.write(args.opath)

