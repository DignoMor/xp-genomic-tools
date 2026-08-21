"""MotifTools random_seq command."""

from __future__ import annotations

from RGTools.MotifGeneration import iter_random_sequences

from .exclusions import resolve_exclusion_inputs
from .output import format_fasta, validate_output_flags, write_text_output


class RandomSeq:
    @staticmethod
    def set_parser(parser):
        parser.add_argument(
            "--sequence_length",
            type=int,
            required=True,
            help="Length of each generated sequence.",
        )
        parser.add_argument(
            "--num_sequences",
            type=int,
            required=True,
            help="Number of sequences to generate.",
        )
        parser.add_argument(
            "--alphabet",
            default="ACGT",
            help="Ordered alphabet for uniform sampling (default ACGT).",
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
            "--seed",
            type=int,
            help="Deterministic random seed (0 is valid).",
        )
        parser.add_argument(
            "--max_attempts",
            type=int,
            default=10000,
            help="Maximum candidate attempts per output when exclusions are used.",
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

        sequences = iter_random_sequences(
            args.sequence_length,
            args.num_sequences,
            alphabet=args.alphabet,
            seed=args.seed,
            meme=meme,
            exclusions=exclusions,
            max_attempts=args.max_attempts,
        )
        records = [
            (f"random_seq_{index}", sequence)
            for index, sequence in enumerate(sequences)
        ]
        write_text_output(format_fasta(records), args.output, force=args.force)
