import argparse

from .cli import GenomicElementTools


def main(argv=None):
    parser = argparse.ArgumentParser(description="Genomic element tools.")
    GenomicElementTools.set_parser(parser)
    args = parser.parse_args(argv)
    GenomicElementTools.main(args)


if __name__ == "__main__":
    main()
