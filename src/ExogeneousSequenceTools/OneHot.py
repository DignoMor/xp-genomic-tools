import argparse

from RGTools.ExogeneousSequences import ExogeneousSequences


class OneHot:
    @staticmethod
    def set_parser_onehot(parser: argparse.ArgumentParser):
        ExogeneousSequences.set_parser_exogeneous_sequences(parser)
        parser.add_argument(
            "--opath",
            help="Output path for the one-hot encoded sequence.",
            type=str,
            required=True,
        )

    @staticmethod
    def onehot_main(args: argparse.Namespace):
        exogeneous_sequences = ExogeneousSequences(args.fasta)
        output_arr = exogeneous_sequences.get_all_region_one_hot()
        output_arr = output_arr.transpose(0, 2, 1)
        exogeneous_sequences.load_region_array_from_arr("onehot", output_arr)
        exogeneous_sequences.save_anno_npy("onehot", args.opath)
