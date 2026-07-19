import argparse

from .cli import ExogeneousSequenceTools


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exogeneous sequence tools.")
    ExogeneousSequenceTools.set_parser(parser)
    args = parser.parse_args(argv)
    ExogeneousSequenceTools.main(args)


if __name__ == "__main__":
    main()
