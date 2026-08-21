"""Motif-centric CLI for motif generation and transformation."""

from __future__ import annotations

import argparse
import sys

from .cli import MotifTools


def main(argv=None):
    parser = argparse.ArgumentParser(description="Motif generation and transformation tools.")
    MotifTools.set_parser(parser)
    args = parser.parse_args(argv)
    try:
        MotifTools.main(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except OSError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
