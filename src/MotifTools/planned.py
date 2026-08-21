"""Planned MotifTools commands (not yet implemented)."""

from __future__ import annotations


def _planned(subcommand: str) -> None:
    raise ValueError(f"{subcommand} is planned but not yet implemented.")

class Barcodes:
    @staticmethod
    def set_parser(parser):
        parser.add_argument("--barcode_length", type=int, required=True)
        parser.add_argument("--alphabet", default="ACGT")
        parser.add_argument("--motif_file")
        parser.add_argument("--exclude", action="append")
        parser.add_argument("--max_candidates", type=int, default=1000000)
        parser.add_argument("--output", required=True)
        parser.add_argument("--force", action="store_true", default=False)

    @staticmethod
    def main(args):
        _planned("barcodes")
