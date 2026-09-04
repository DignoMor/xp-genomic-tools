import argparse

from .cli import ExogenousSequenceTools


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exogenous sequence tools.")
    ExogenousSequenceTools.set_parser(parser)
    args = parser.parse_args(argv)
    ExogenousSequenceTools.main(args)


if __name__ == "__main__":
    main()
