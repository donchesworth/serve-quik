from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Optional

KWARGS = ["source", "target"]


def parse_args(arglist: Optional[list] = None) -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--project_name", required=True)
    parser.add_argument("-mt", "--model_type")
    parser.add_argument("-src", "--source")
    parser.add_argument("-tgt", "--target")
    parser.add_argument("-hdlr", "--handler")
    args = parser.parse_args(arglist)
    args.kwargs = {k: vars(args)[k] for k in KWARGS if vars(args)[k]}
    return args
