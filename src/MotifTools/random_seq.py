"""MotifTools random_seq command."""

from __future__ import annotations

from RGTools.MotifGeneration import iter_random_sequences

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
            help="MEME motif collection for exclusions (not yet delivered).",
        )
        parser.add_argument(
            "--exclude",
            action="append",
            help="Motif exclusion MOTIF=CUTOFF (not yet delivered).",
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
        if args.motif_file is not None:
            raise ValueError(
                "Motif exclusion for random_seq is not yet delivered; "
                "omit --motif_file for unconstrained generation."
            )
        if args.exclude:
            raise ValueError(
                "Motif exclusion for random_seq is not yet delivered; "
                "omit --exclude for unconstrained generation."
            )

        sequences = iter_random_sequences(
            args.sequence_length,
            args.num_sequences,
            alphabet=args.alphabet,
            seed=args.seed,
        )
        records = [
            (f"random_seq_{index}", sequence)
            for index, sequence in enumerate(sequences)
        ]
        write_text_output(format_fasta(records), args.output, force=args.force)
