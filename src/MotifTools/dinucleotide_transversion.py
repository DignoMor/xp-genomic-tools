"""MotifTools dinucleotide_transversion command."""

from __future__ import annotations

import math
import sys

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
        parser.add_argument(
            "--warn_score_cutoff",
            type=float,
            default=0,
            help=(
                "Heuristic source-motif score cutoff (finite real; default 0). "
                "Warn on stderr if the full-width target scores at or above this "
                "value on either strand. Diagnostics only; does not change FASTA."
            ),
        )

    @staticmethod
    def main(args):
        validate_output_flags(args.output, args.force)
        cutoff = args.warn_score_cutoff
        if not math.isfinite(cutoff):
            raise ValueError(
                "--warn_score_cutoff must be a finite real number, "
                f"found {cutoff!r}."
            )
        meme = MemeMotif(args.motif_file)
        sequence = generate_dinucleotide_transversion(meme, args.motif_name)
        _emit_source_motif_warning(meme, args.motif_name, sequence, cutoff)
        records = [
            (f"dinucleotide_transversion_{args.motif_name}", sequence),
        ]
        write_text_output(format_fasta(records), args.output, force=args.force)


def _emit_source_motif_warning(meme, motif_name, sequence, cutoff):
    alphabet = meme.get_alphabet()
    pwm = meme.get_motif_pwm(motif_name)
    background = meme.get_bg_freq()
    forward = MemeMotif.calculate_pwm_score(
        sequence,
        pwm,
        alphabet,
        background,
        reverse_complement=False,
    )
    reverse = MemeMotif.calculate_pwm_score(
        sequence,
        pwm,
        alphabet,
        background,
        reverse_complement=True,
    )
    if forward >= cutoff or reverse >= cutoff:
        print(
            f"Warning: motif {motif_name} scores {float(forward):.10g} (forward) "
            f"and {float(reverse):.10g} (reverse complement) against the source "
            f"PWM; cutoff {float(cutoff):.10g}.",
            file=sys.stderr,
        )
