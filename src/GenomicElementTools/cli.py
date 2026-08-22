#!/usr/bin/env python

import argparse

from .count_bw import CountSingleBw, CountPairedBw
from .pad_region import PadRegion
from .bed2tss_bed import Bed2TssBed
from .one_hot import OneHot
from .motif_search import MotifSearch
from .filter_motif_score import FilterMotifScore
from .track2tss_bed import Track2TssBed
from .export import GenomicElementExport
from .GenomicElementImport import GenomicElementImport
from .get_context_ge import GetContextGe
from .mask_op import MaskOp
from .select_tss_relative_track import SelectTssRelativeTrack


class GenomicElementTools:
    @staticmethod
    def set_parser(parser):
        subparsers = parser.add_subparsers(dest="subcommand")

        parser_count_single_bw = subparsers.add_parser("count_single_bw",
                                                       help="Count signal in a single bigwig files.",
                                                       )

        CountSingleBw.set_parser(parser_count_single_bw)

        parser_count_paired_bw = subparsers.add_parser("count_paired_bw",
                                                       help="Count signal in a paired bigwig files.",
                                                       )

        CountPairedBw.set_parser(parser_count_paired_bw)

        parser_pad_region = subparsers.add_parser("pad_region",
                                                  help="Pad regions. This program differs from "
                                                       "padding_bed.py in that it conserve the "
                                                       "order of elements in Genomic Elements files.",
                                                  )

        PadRegion.set_parser(parser_pad_region)

        parser_bed2tssbed = subparsers.add_parser("bed2tssbed",
                                                  help="Convert bed file to TSS bed file.",
                                                  )

        Bed2TssBed.set_parser(parser_bed2tssbed)

        parser_onehot = subparsers.add_parser("onehot",
                                              help="One-hot encode the sequence. Only support elements of the same size.",
                                              )

        OneHot.set_parser(parser_onehot)

        parser_motif_search = subparsers.add_parser("motif_search",
                                                  help="Search for motifs in sequences.",
                                                  )
        MotifSearch.set_parser(parser_motif_search)

        parser_track2tss_bed = subparsers.add_parser("track2tss_bed",
                                                    help="Produce TSS bed file from track.",
                                                    )

        Track2TssBed.set_parser(parser_track2tss_bed)

        parser_filter_motif_score = subparsers.add_parser("filter_motif_score",
                                                          help="Filter motif search scores.",
                                                          )

        FilterMotifScore.set_parser(parser_filter_motif_score)

        parser_export = subparsers.add_parser("export",
                                              help="Export to other formats.",
                                              )

        GenomicElementExport.set_parser(parser_export)

        parser_import = subparsers.add_parser("import",
                                              help="Import from other formats.",
                                              )

        GenomicElementImport.set_parser(parser_import)

        parser_get_context_ge = subparsers.add_parser("get_context_ge",
                                                      help="Generate context windows around regions.",
                                                      )

        GetContextGe.set_parser(parser_get_context_ge)

        parser_mask_op = subparsers.add_parser(
            "mask_op",
            help="Apply logical operations to mask arrays.",
        )

        MaskOp.set_parser(parser_mask_op)

        parser_select_tss_relative_track = subparsers.add_parser(
            "select_tss_relative_track",
            help="Select a TSS-relative track score and emit coordinate/mask annotations.",
        )

        SelectTssRelativeTrack.set_parser(parser_select_tss_relative_track)

    @staticmethod
    def main(args):
        if args.subcommand == "count_single_bw":
            CountSingleBw.main(args)
        elif args.subcommand == "count_paired_bw":
            CountPairedBw.main(args)
        elif args.subcommand == "pad_region":
            PadRegion.main(args)
        elif args.subcommand == "bed2tssbed":
            Bed2TssBed.main(args)
        elif args.subcommand == "onehot":
            OneHot.main(args)
        elif args.subcommand == "motif_search":
            MotifSearch.main(args)
        elif args.subcommand == "track2tss_bed":
            Track2TssBed.main(args)
        elif args.subcommand == "filter_motif_score":
            FilterMotifScore.main(args)
        elif args.subcommand == "export":
            GenomicElementExport.main(args)
        elif args.subcommand == "import":
            GenomicElementImport.main(args)
        elif args.subcommand == "get_context_ge":
            GetContextGe.main(args)
        elif args.subcommand == "mask_op":
            MaskOp.main(args)
        elif args.subcommand == "select_tss_relative_track":
            SelectTssRelativeTrack.main(args)
        else:
            raise ValueError("Unknown subcommand: {}".format(args.subcommand))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genomic element tools.")
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args()
    GenomicElementTools.main(args)
