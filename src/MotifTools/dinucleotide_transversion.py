"""MotifTools dinucleotide_transversion command."""

from __future__ import annotations

from RGTools import MemeMotif
from RGTools.MotifGeneration import generate_dinucleotide_transversion

from .output import format_fasta, validate_output_flags, write_text_output


class DinucleotideTransversion:
    @staticmethod
    def set_parser(parser):
        parser.add_argument(
            "--motif_file",
            required=True,
            help="Input MEME motif collection file.",
        )
        parser.add_argument(
            "--motif_name",
            required=True,
            help="Name of the motif used to derive the transversion target.",
        )
        parser.add_argument(
            "--output",
            required=True,
            help="Output FASTA path, or '-' for stdout.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Replace an existing output file.",
        )

    @staticmethod
    def main(args):
        validate_output_flags(args.output, args.force)
        meme = MemeMotif(args.motif_file)
        sequence = generate_dinucleotide_transversion(meme, args.motif_name)
        records = [
            (f"dinucleotide_transversion_{args.motif_name}", sequence),
        ]
        write_text_output(format_fasta(records), args.output, force=args.force)
