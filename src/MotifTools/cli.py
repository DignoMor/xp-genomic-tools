"""MotifTools CLI dispatcher."""

from __future__ import annotations

import argparse

from .anti_motif import AntiMotif
from .barcodes import Barcodes
from .pwm_seq import PwmSeq
from .random_seq import RandomSeq


class MotifTools:
    @staticmethod
    def set_parser(parser):
        subparsers = parser.add_subparsers(dest="subcommand", required=True)

        parser_random = subparsers.add_parser(
            "random_seq",
            help="Generate random sequences.",
        )
        RandomSeq.set_parser(parser_random)

        parser_pwm = subparsers.add_parser(
            "pwm_seq",
            help="Sample sequences from one PWM.",
        )
        PwmSeq.set_parser(parser_pwm)

        parser_barcodes = subparsers.add_parser(
            "barcodes",
            help="Enumerate motif-filtered barcodes.",
        )
        Barcodes.set_parser(parser_barcodes)

        parser_anti = subparsers.add_parser(
            "anti_motif",
            help="Derive anti-motifs from a MEME collection.",
        )
        AntiMotif.set_parser(parser_anti)

    @staticmethod
    def main(args):
        if args.subcommand == "random_seq":
            RandomSeq.main(args)
        elif args.subcommand == "pwm_seq":
            PwmSeq.main(args)
        elif args.subcommand == "barcodes":
            Barcodes.main(args)
        elif args.subcommand == "anti_motif":
            AntiMotif.main(args)
        else:
            raise ValueError(f"Unknown subcommand: {args.subcommand}")
