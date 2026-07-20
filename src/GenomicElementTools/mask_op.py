import numpy as np
from RGTools.GenomicElements import GenomicElements


class MaskOp:
    @staticmethod
    def set_parser(parser):
        subparsers = parser.add_subparsers(dest="operation")

        parser_intersect = subparsers.add_parser(
            "intersect",
            help="Element-wise AND across input mask arrays.",
        )
        GenomicElements.set_parser_genomic_element_region(parser_intersect)
        parser_intersect.add_argument(
            "--mask_npy",
            help="Path to input mask array (.npy or single-array .npz). Use multiple times.",
            action="append",
            required=True,
            type=str,
        )
        parser_intersect.add_argument(
            "--opath",
            help="Output path for the resulting mask array (.npy).",
            required=True,
            type=str,
        )

        parser_union = subparsers.add_parser(
            "union",
            help="Element-wise OR across input mask arrays.",
        )
        GenomicElements.set_parser_genomic_element_region(parser_union)
        parser_union.add_argument(
            "--mask_npy",
            help="Path to input mask array (.npy or single-array .npz). Use multiple times.",
            action="append",
            required=True,
            type=str,
        )
        parser_union.add_argument(
            "--opath",
            help="Output path for the resulting mask array (.npy).",
            required=True,
            type=str,
        )

        parser_opposite = subparsers.add_parser(
            "opposite",
            help="Element-wise NOT for one input mask array.",
        )
        GenomicElements.set_parser_genomic_element_region(parser_opposite)
        parser_opposite.add_argument(
            "--mask_npy",
            help="Path to input mask array (.npy or single-array .npz).",
            required=True,
            type=str,
        )
        parser_opposite.add_argument(
            "--opath",
            help="Output path for the resulting mask array (.npy).",
            required=True,
            type=str,
        )

    @staticmethod
    def main_intersect(args):
        ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=None,
        )
        mask_arr_list = []
        for i, mask_path in enumerate(args.mask_npy):
            mask_name = f"__mask_{i}"
            ge.load_region_anno_from_npy(mask_name, mask_path, anno_type="mask")
            mask_arr_list.append(ge.get_mask_arr(mask_name).reshape(-1,))

        if len(mask_arr_list) < 2:
            raise ValueError("At least two --mask_npy inputs are required.")
        out_arr = np.logical_and.reduce(mask_arr_list)
        ge.load_mask_from_arr("__mask_op_result__", out_arr)
        ge.save_anno_npy("__mask_op_result__", args.opath)

    @staticmethod
    def main_union(args):
        ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=None,
        )
        mask_arr_list = []
        for i, mask_path in enumerate(args.mask_npy):
            mask_name = f"__mask_{i}"
            ge.load_region_anno_from_npy(mask_name, mask_path, anno_type="mask")
            mask_arr_list.append(ge.get_mask_arr(mask_name).reshape(-1,))

        if len(mask_arr_list) < 2:
            raise ValueError("At least two --mask_npy inputs are required.")
        out_arr = np.logical_or.reduce(mask_arr_list)
        ge.load_mask_from_arr("__mask_op_result__", out_arr)
        ge.save_anno_npy("__mask_op_result__", args.opath)

    @staticmethod
    def main_opposite(args):
        ge = GenomicElements(
            region_file_path=args.region_file_path,
            region_file_type=args.region_file_type,
            fasta_path=None,
        )
        ge.load_region_anno_from_npy("__mask_0", args.mask_npy, anno_type="mask")
        out_arr = np.logical_not(ge.get_mask_arr("__mask_0"))
        ge.load_mask_from_arr("__mask_op_result__", out_arr)
        ge.save_anno_npy("__mask_op_result__", args.opath)

    @staticmethod
    def main(args):
        if args.operation == "intersect":
            MaskOp.main_intersect(args)
        elif args.operation == "union":
            MaskOp.main_union(args)
        elif args.operation == "opposite":
            MaskOp.main_opposite(args)
        else:
            raise ValueError(f"Unknown mask_op operation: {args.operation}")
