import argparse

from RGTools.ExogenousSequences import ExogenousSequences


class OneHot:
    @staticmethod
    def set_parser_onehot(parser: argparse.ArgumentParser):
        ExogenousSequences.set_parser_exogenous_sequences(parser)
        parser.add_argument(
            "--opath",
            help="Output path for the one-hot encoded sequence.",
            type=str,
            required=True,
        )

    @staticmethod
    def onehot_main(args: argparse.Namespace):
        exogenous_sequences = ExogenousSequences(args.fasta)
        output_arr = exogenous_sequences.get_all_region_one_hot()
        output_arr = output_arr.transpose(0, 2, 1)
        exogenous_sequences.load_region_array_from_arr("onehot", output_arr)
        exogenous_sequences.save_anno_npy("onehot", args.opath)
