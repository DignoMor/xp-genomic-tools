"""MotifTools anti_motif command."""

from __future__ import annotations

from RGTools import MemeMotif
from RGTools.MotifGeneration import make_anti_motifs

from .output import validate_output_flags, write_meme_output


class AntiMotif:
    @staticmethod
    def set_parser(parser):
        parser.add_argument(
            "--motif_file",
            required=True,
            help="Input MEME motif collection file.",
        )
        parser.add_argument(
            "--output",
            required=True,
            help="Output MEME path, or '-' for stdout.",
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
        source = MemeMotif(args.motif_file)
        anti = make_anti_motifs(source)
        write_meme_output(anti, args.output, force=args.force)
