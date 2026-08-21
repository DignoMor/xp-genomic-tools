"""MotifTools pwm_seq command."""

from __future__ import annotations

from RGTools import MemeMotif
from RGTools.MotifGeneration import iter_pwm_sequences

from .output import format_fasta, validate_output_flags, write_text_output


class PwmSeq:
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
            help="Name of the motif to sample.",
        )
        parser.add_argument(
            "--num_sequences",
            type=int,
            required=True,
            help="Number of sequences to generate.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Deterministic random seed (0 is valid).",
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
        sequences = iter_pwm_sequences(
            meme,
            args.motif_name,
            args.num_sequences,
            seed=args.seed,
        )
        records = [
            (f"pwm_{args.motif_name}_{index}", sequence)
            for index, sequence in enumerate(sequences)
        ]
        write_text_output(format_fasta(records), args.output, force=args.force)
