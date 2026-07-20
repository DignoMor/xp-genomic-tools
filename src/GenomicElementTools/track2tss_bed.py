
import pandas as pd
import numpy as np

from RGTools.BedTable import BedRegion
from RGTools.GenomicElements import GenomicElements

class Track2TssBed:
    @staticmethod
    def set_parser(parser):

        GenomicElements.set_parser_genomic_element_region(parser)
        parser.add_argument("--track", 
                            help="The track npy file path.",
                            type=str,
                            required=True,
                            )

        parser.add_argument("--opath",
                            help="Output path for the TSS BED file",
                            type=str,
                            required=True,
                            )

        parser.add_argument("--output_site",
                            help="The site for output. [TSS] ({})".format(
                                ", ".join(Track2TssBed.get_output_site_types()),
                            ),
                            default="MaxAbsSig",
                            type=str,
                            )

    @staticmethod
    def get_output_site_types():
        return ["MaxAbsSig"]

    @staticmethod
    def get_output_site_coord(region: BedRegion, 
                              track: "np.ndarray", 
                              output_site_type: str, 
                              ):

        if output_site_type == "MaxAbsSig":
            ouput_arg = np.argmax(np.abs(track))
        else:
            raise ValueError("Unknown output site type: {}".format(output_site_type))
        
        return region["start"] + ouput_arg

    @staticmethod
    def main(args):

        genomic_elements = GenomicElements(region_file_path=args.region_file_path,
                                           region_file_type=args.region_file_type,
                                           fasta_path=None, 
                                           )
        genomic_elements.load_region_anno_from_npy("track", args.track, anno_type="track")
        track_list = genomic_elements.get_track_list("track")

        region_bt = genomic_elements.get_region_bed_table()
        output_dict_list = []

        for i, region in enumerate(region_bt.iter_regions()):
            track = track_list[i]

            output_site = Track2TssBed.get_output_site_coord(region, track, args.output_site)
            output_region = region.to_dict()
            output_region["start"] = output_site
            output_region["end"] = output_site + 1

            output_dict_list.append(output_region)

        output_region_bt = region_bt._clone_empty()
        output_region_bt.load_from_dataframe(pd.DataFrame(output_dict_list,
                                                          columns=region_bt.column_names,
                                                          ))
        
        output_region_bt.write(args.opath)