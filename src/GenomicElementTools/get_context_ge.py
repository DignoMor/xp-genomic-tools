
from RGTools.GenomicElements import GenomicElements


class GetContextGe:
    @staticmethod
    def set_parser(parser):
        method_subparsers = parser.add_subparsers(dest="method")

        nearest_parser = method_subparsers.add_parser(
            "nearest",
            help="Select nearest context region by closest-end distance.",
        )

        GenomicElements.set_parser_genomic_element_region(nearest_parser)
        nearest_parser.add_argument(
            "--context_file_path",
            help="Path to the context region file.",
            required=True,
            type=str,
        )
        nearest_parser.add_argument(
            "--context_file_type",
            help="Type of the context region file. "
                 "Valid types: {}".format(
                     list(GenomicElements.get_region_file_suffix2class_dict().keys())
                 ),
            required=True,
            type=str,
            choices=GenomicElements.get_region_file_suffix2class_dict().keys(),
        )
        nearest_parser.add_argument(
            "--opath",
            help="Path to the output region file.",
            required=True,
            type=str,
        )

        windowed_argmax_parser = method_subparsers.add_parser(
            "windowed_argmax",
            help="Select context region with maximum provided stat in each input window.",
        )
        GenomicElements.set_parser_genomic_element_region(windowed_argmax_parser)
        windowed_argmax_parser.add_argument(
            "--context_file_path",
            help="Path to the context region file.",
            required=True,
            type=str,
        )
        windowed_argmax_parser.add_argument(
            "--context_file_type",
            help="Type of the context region file. "
                 "Valid types: {}".format(
                     list(GenomicElements.get_region_file_suffix2class_dict().keys())
                 ),
            required=True,
            type=str,
            choices=GenomicElements.get_region_file_suffix2class_dict().keys(),
        )
        windowed_argmax_parser.add_argument(
            "--context_stat_path",
            help="Path to the context stat .npy file.",
            required=True,
            type=str,
        )
        windowed_argmax_parser.add_argument(
            "--opath",
            help="Path to the output region file.",
            required=True,
            type=str,
        )

    @staticmethod
    def _closest_end_distance(region, context_region):
        # Minimal absolute distance between any pair of interval ends.
        return min(
            abs(region["start"] - context_region["start"]),
            abs(region["start"] - context_region["end"]),
            abs(region["end"] - context_region["start"]),
            abs(region["end"] - context_region["end"]),
        )

    @staticmethod
    def _run_nearest(args):
        input_ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=None,
        )
        context_ge = GenomicElements(
            region_file_path=args.context_file_path,
            region_file_type=args.context_file_type,
            fasta_path=None,
        )

        input_regions = list(input_ge.get_region_bed_table().iter_regions())
        context_bt = context_ge.get_region_bed_table()
        context_regions = list(context_bt.iter_regions())

        context_by_chrom = {}
        for context_region in context_regions:
            context_by_chrom.setdefault(context_region["chrom"], []).append(context_region)

        output_regions = []
        for region in input_regions:
            same_chrom_contexts = context_by_chrom.get(region["chrom"], [])
            if len(same_chrom_contexts) == 0:
                raise ValueError(
                    f"No context regions found on chromosome '{region['chrom']}' "
                    f"for input region {region['chrom']}:{region['start']}-{region['end']}."
                )

            nearest_context = min(
                same_chrom_contexts,
                key=lambda context_region: GetContextGe._closest_end_distance(region, context_region),
            )
            output_regions.append(nearest_context)

        output_bt = context_bt._clone_empty()
        output_bt.load_from_bed_regions(output_regions)
        output_bt.write(args.opath)

    @staticmethod
    def _run_windowed_argmax(args):
        input_ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=None,
        )
        context_ge = GenomicElements(
            region_file_path=args.context_file_path,
            region_file_type=args.context_file_type,
            fasta_path=None,
        )
        context_ge.load_region_anno_from_npy("context_stat",
                                             args.context_stat_path,
                                             anno_type="stat",
                                             )

        context_stat = context_ge.get_stat_arr("context_stat")

        input_regions = list(input_ge.get_region_bed_table().iter_regions())
        context_bt = context_ge.get_region_bed_table()
        context_regions = list(context_bt.iter_regions())

        context_by_chrom = {}
        for context_idx, context_region in enumerate(context_regions):
            context_by_chrom.setdefault(context_region["chrom"], []).append((context_idx, context_region))

        output_regions = []
        for window_region in input_regions:
            same_chrom_contexts = context_by_chrom.get(window_region["chrom"], [])

            in_window = []
            for context_idx, context_region in same_chrom_contexts:
                if context_region["start"] >= window_region["start"] and context_region["end"] <= window_region["end"]:
                    in_window.append((context_idx, context_region))

            if len(in_window) == 0:
                raise ValueError(
                    f"No context regions found within input window "
                    f"{window_region['chrom']}:{window_region['start']}-{window_region['end']}."
                )

            # Stable tie-break by earliest context index.
            _, best_context_region = max(
                in_window,
                key=lambda x: context_stat.reshape(-1)[x[0]],
            )
            output_regions.append(best_context_region)

        output_bt = context_bt._clone_empty()
        output_bt.load_from_bed_regions(output_regions)
        output_bt.write(args.opath)

    @staticmethod
    def main(args):
        if args.method == "nearest":
            GetContextGe._run_nearest(args)
        elif args.method == "windowed_argmax":
            GetContextGe._run_windowed_argmax(args)
        else:
            raise ValueError(f"Unknown get_context_ge method: {args.method}")
