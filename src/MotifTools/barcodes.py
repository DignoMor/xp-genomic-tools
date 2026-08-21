"""MotifTools barcodes command."""

from __future__ import annotations

from RGTools.MotifGeneration import iter_barcodes

from .exclusions import resolve_exclusion_inputs
from .output import format_fasta, validate_output_flags, write_text_output


class Barcodes:
    @staticmethod
    def set_parser(parser):
        parser.add_argument(
            "--barcode_length",
            type=int,
            required=True,
            help="Length of each enumerated barcode.",
        )
        parser.add_argument(
            "--alphabet",
            default="ACGT",
            help="Ordered alphabet for Cartesian enumeration (default ACGT).",
        )
        parser.add_argument(
            "--motif_file",
            help="MEME motif collection for exclusions.",
        )
        parser.add_argument(
            "--exclude",
            action="append",
            help="Motif exclusion MOTIF=CUTOFF (repeatable).",
        )
        parser.add_argument(
            "--max_candidates",
            type=int,
            default=1000000,
            help="Maximum pre-exclusion candidate count before enumeration.",
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
        meme, exclusions = resolve_exclusion_inputs(args.motif_file, args.exclude)

        barcodes = iter_barcodes(
            args.barcode_length,
            alphabet=args.alphabet,
            meme=meme,
            exclusions=exclusions,
            max_candidates=args.max_candidates,
        )
        records = [
            (f"barcode_{index}", barcode)
            for index, barcode in enumerate(barcodes)
        ]
        write_text_output(format_fasta(records), args.output, force=args.force)
