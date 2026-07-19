
import pandas as pd

from RGTools.GenomicElements import GenomicElements

class Bed2TssBed:
    @staticmethod
    def set_parser(parser):
        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--opath",
                            help="Output path for the TSS BED file",
                            type=str,
                            required=True,
                            )

        parser.add_argument("--output_site", 
                            help="The site for output. [TSS] ({})".format(
                                ", ".join(Bed2TssBed.get_output_site_types()), 
                            ), 
                            default="TSS",
                            type=str, 
                            )

    @staticmethod
    def get_output_site_types():
        return ["TSS", "center"]

    @staticmethod
    def get_output_site_coord(region, output_site_type):
        if output_site_type == "TSS":
            if region["strand"] == "+":
                site = region["start"]
            elif region["strand"] == "-":
                site = region["end"] - 1
        elif output_site_type == "center":
            site = int((region["start"] + region["end"]) // 2)
        
        return site

    @staticmethod
    def main(args):
        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=None, 
                                           )
        region_bt = genomic_elements.get_region_bed_table()

        output_region_list = []
        for region in region_bt.iter_regions():
            output_coord = Bed2TssBed.get_output_site_coord(region, args.output_site)
            
            output_region_dict = region.to_dict()
            output_region_dict["start"] = output_coord
            output_region_dict["end"] = output_coord + 1
            output_region_list.append(output_region_dict)

        output_df = pd.DataFrame(output_region_list, 
                                 columns=region_bt.column_names, 
                                 )
        
        output_bt = region_bt._clone_empty()
        output_bt.load_from_dataframe(output_df)
        output_bt.write(args.opath)

