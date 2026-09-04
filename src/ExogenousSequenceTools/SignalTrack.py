
import sys

import numpy as np

from RGTools.ExogenousSequences import ExogenousSequences

class SignalTrack:
    @staticmethod
    def _set_track_dim_reduction_common_parser_args(parser):

        parser.add_argument("--input_npy",
                            help="Path to the signal track npy file.",
                            required=True,
                            )

        parser.add_argument("--output_npy",
                            help="Path to the output stat npy file.",
                            required=True,
                            )
        
        parser.add_argument("--search_range",
                            help="Search range for the signal track. Format: 'start,end'. "
                                 "If not provided, the whole signal track will be used. "
                                 "Range follows half-open and 0-index convention.",
                            default=None, 
                            type=str,
                            )

    @staticmethod
    def set_parser_track_dim_reduction(parser):
        """Set up parser for track dimension reduction operations."""
        subparsers = parser.add_subparsers(dest="operation", required=True)
        
        parser_max = subparsers.add_parser("max")
        SignalTrack._set_track_dim_reduction_common_parser_args(parser_max)
        
        parser_argmax = subparsers.add_parser("argmax")
        SignalTrack._set_track_dim_reduction_common_parser_args(parser_argmax)
        
        parser_min = subparsers.add_parser("min")
        SignalTrack._set_track_dim_reduction_common_parser_args(parser_min)
        
        parser_argmin = subparsers.add_parser("argmin")
        SignalTrack._set_track_dim_reduction_common_parser_args(parser_argmin)

    @staticmethod
    def _track_dim_reduction(args, operation):
        signal_track = np.load(args.input_npy)

        if args.search_range: 
            start, end = map(int, args.search_range.split(","))
            signal_track[:, :start] = -np.inf
            signal_track[:, end:] = -np.inf

        if operation == "argmax":
            output_stat = signal_track.argmax(axis=1, keepdims=True)
        elif operation == "max":
            output_stat = signal_track.max(axis=1, keepdims=True)
        elif operation == "argmin":
            output_stat = signal_track.argmin(axis=1, keepdims=True)
        elif operation == "min":
            output_stat = signal_track.min(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        np.save(args.output_npy, output_stat)

    @staticmethod
    def track_dim_reduction_main(args):
        SignalTrack._track_dim_reduction(args, args.operation)
    
    @staticmethod
    def set_parser_gen_track(parser):
        operation_parser = parser.add_subparsers(dest="operation", required=True)
        parser_single_loc = operation_parser.add_parser("single_loc")
        ExogenousSequences.set_parser_exogenous_sequences(parser_single_loc)
        parser_single_loc.add_argument("--loc", 
                                       help="Location to generate the track for.",
                                       required=True,
                                       type=int,
                                       )
        parser_single_loc.add_argument("--output_npy",
                                       help="Path to the output stat npy file.",
                                       required=True,
                                       )

    @staticmethod
    def _gen_track(args, operation):
        if operation == "single_loc":
            es = ExogenousSequences(args.fasta)

            signal_track = np.ones((es.get_num_regions(), 1), 
                                   dtype=np.int64, 
                                   ) * args.loc
            es.load_region_stat_from_arr("loc", signal_track)
            es.save_anno_npy("loc", args.output_npy)

    @staticmethod
    def gen_track_main(args):
        SignalTrack._gen_track(args, args.operation)

    @staticmethod
    def set_parser_print_stat(parser):
        parser.add_argument("--input_npy",
                            help="Path to the region stat npy file.",
                            required=True,
                            )
    
    @staticmethod
    def print_stat_main(args):
        region_stat = np.load(args.input_npy)

        if not region_stat.shape[1]== 1:
            raise ValueError("The second dimension of the region stat must be 1.")

        for s in region_stat[:, 0]:
            print(s)
